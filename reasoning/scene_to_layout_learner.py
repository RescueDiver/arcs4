# reasoning/scene_to_layout_learner.py

"""
Scene-to-layout learner.

Purpose:
    Learn the direct symbolic correlation between the whole input scene
    and the whole output marker layout.

This file does NOT hardcode task ids.
This file does NOT force exact test answers.
This file learns from train examples only.

Main idea:
    input scene signature
        outside_all
        outer_1blobs_below
        inner_1blobs_center

    output layout signature
        outside_right
        outer_below
        inner_center

The existing marker-placement learner handles:
    one source fact -> one marker target

This file adds:
    whole source fact set -> whole output marker set
"""


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def sorted_unique(items):
    result = []

    for item in items:
        if item is None:
            continue

        if item not in result:
            result.append(item)

    result.sort()
    return result


def source_fact_to_abstract_key(source):
    """
    Convert a rich source fact into a reusable abstract scene key.

    Examples:
        outer_1blobs_right -> outer_anyblobs_right
        inner_2blobs_left  -> inner_anyblobs_left
        single_2blobs_above -> single_anyblobs_above
        outside_all -> outside_all
    """

    container_role = source.get("container_role")
    direction = source.get("direction")

    if container_role == "outside_all":
        return "outside_all"

    if container_role in ("outer", "inner", "single"):
        if direction in ("center", "above", "below", "left", "right"):
            return f"{container_role}_anyblobs_{direction}"

    return None


def source_fact_to_simple_key(source):
    """
    Use the source's simple key when available.

    Examples:
        outer_right
        inner_center
        single_above
        outside_all
    """

    return source.get("simple_source_key")


def marker_to_target_key(marker):
    """
    Convert a symbolic marker description into a target key.

    Examples:
        inner_center
        outer_right
        outside_right
    """

    if marker.get("shape_mismatch"):
        return None

    return marker.get("symbolic_position")


def scene_signature_from_source_facts(source_facts):
    """
    Build a full scene signature from input-side facts.

    Returns both:
        exact signature:
            outer_2blobs_left
            outer_2blobs_above
            inner_1blobs_center

        abstract signature:
            outer_anyblobs_left
            outer_anyblobs_above
            inner_anyblobs_center
    """

    exact = []
    simple = []
    abstract = []

    for source in source_facts:
        exact.append(source.get("source_key"))
        simple.append(source_fact_to_simple_key(source))
        abstract.append(source_fact_to_abstract_key(source))

    return {
        "exact": tuple(sorted_unique(exact)),
        "simple": tuple(sorted_unique(simple)),
        "abstract": tuple(sorted_unique(abstract)),
    }


def layout_signature_from_markers(symbolic_marker_positions):
    """
    Build a target layout signature from output-side marker facts.
    """

    targets = []

    for marker in symbolic_marker_positions:
        target = marker_to_target_key(marker)

        if target is not None:
            targets.append(target)

    return tuple(sorted_unique(targets))


# ============================================================
# TRANSLATION EVIDENCE
# ============================================================

def expected_target_for_source(source):
    """
    Predict the natural target key for a source fact.

    This is not a final answer.
    It is a symbolic hypothesis used for learning evidence.

    Examples:
        inner center -> inner_center
        outer right  -> outer_right
        outside_all  -> outside_right
        single above -> outer_above
    """

    role = source.get("container_role")
    direction = source.get("direction")

    if role == "outside_all":
        return "outside_right"

    if direction not in ("center", "above", "below", "left", "right"):
        return None

    if role == "inner":
        return f"inner_{direction}"

    if role == "outer":
        return f"outer_{direction}"

    if role == "single":
        return f"outer_{direction}"

    return None


def add_count(bucket, key):
    if key is None:
        return

    bucket[key] = bucket.get(key, 0) + 1


def learn_scene_to_layout_rule(learning_examples):
    """
    Learn whole-scene to whole-layout relationships.

    Learns:
        1. exact scene templates
        2. abstract scene templates
        3. source-key -> target-key translation evidence
        4. global target layout habits
    """

    exact_templates = {}
    abstract_templates = {}
    translation_evidence = {}
    global_target_counts = {}

    examples = []

    def add_translation(source_key, target_key):
        if source_key is None:
            return

        if target_key is None:
            return

        if source_key not in translation_evidence:
            translation_evidence[source_key] = {}

        add_count(translation_evidence[source_key], target_key)

    for example in learning_examples:
        source_facts = example.get("source_facts", [])
        symbolic_markers = example.get("symbolic_marker_positions", [])

        scene_signature = scene_signature_from_source_facts(source_facts)
        layout_signature = layout_signature_from_markers(symbolic_markers)

        if not layout_signature:
            continue

        exact_key = scene_signature["exact"]
        abstract_key = scene_signature["abstract"]

        exact_templates[exact_key] = layout_signature
        abstract_templates[abstract_key] = layout_signature

        for target in layout_signature:
            add_count(global_target_counts, target)

        layout_set = set(layout_signature)

        for source in source_facts:
            natural_target = expected_target_for_source(source)

            if natural_target in layout_set:
                add_translation(source.get("source_key"), natural_target)
                add_translation(source.get("simple_source_key"), natural_target)
                add_translation(source_fact_to_abstract_key(source), natural_target)

        examples.append(
            {
                "scene_exact": exact_key,
                "scene_abstract": abstract_key,
                "layout": layout_signature,
            }
        )

    learned_translations = {}

    for source_key, target_counts in translation_evidence.items():
        best_target = max(
            target_counts,
            key=lambda key: target_counts[key],
        )

        learned_translations[source_key] = {
            "best_target": best_target,
            "target_counts": target_counts,
            "ambiguous": len(target_counts) > 1,
        }

    return {
        "family": "scene_to_layout",
        "mode": "whole_scene_signature_to_output_layout_signature",
        "exact_templates": exact_templates,
        "abstract_templates": abstract_templates,
        "learned_translations": learned_translations,
        "global_target_counts": global_target_counts,
        "examples": examples,
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_layout_targets_for_scene(source_facts, scene_layout_rule):
    """
    Predict output layout targets from input-side symbolic source facts.

    Safe learning priority:

        1. Exact whole-scene example match
        2. Abstract whole-scene example match
        3. Per-source learned translations
        4. Natural fallback for sources without learned translations

    Important:
        Do not return a partial learned translation list too early.
        A partial list can delete valid markers from the scene.
    """

    if not source_facts:
        return []

    if scene_layout_rule is None:
        return []

    def sorted_unique(items):
        result = []

        for item in items:
            if item is None:
                continue

            if item not in result:
                result.append(item)

        return sorted(result)

    def source_fact_to_abstract_key_local(source_key):
        if source_key is None:
            return None

        parts = source_key.split("_")

        if len(parts) < 3:
            return source_key

        role = parts[0]

        if role not in ("single", "inner", "outer"):
            return source_key

        blob_part = parts[1]

        if not blob_part.endswith("blobs"):
            return source_key

        direction = "_".join(parts[2:])

        return f"{role}_anyblobs_{direction}"

    def exact_scene_signature_from_facts(facts):
        return tuple(
            sorted(
                fact.get("source_key")
                for fact in facts
                if fact.get("source_key") is not None
            )
        )

    def abstract_scene_signature_from_facts(facts):
        return tuple(
            sorted(
                source_fact_to_abstract_key_local(fact.get("source_key"))
                for fact in facts
                if fact.get("source_key") is not None
            )
        )

    def get_examples(rule):
        examples = rule.get("examples")

        if isinstance(examples, list):
            return examples

        return []

    def get_example_signature(example, key_options):
        for key in key_options:
            value = example.get(key)

            if value is None:
                continue

            if isinstance(value, tuple):
                return value

            if isinstance(value, list):
                return tuple(value)

        return None

    def get_example_layout(example):
        for key in (
            "layout",
            "layout_signature",
            "target_layout",
            "output_layout",
        ):
            value = example.get(key)

            if value is None:
                continue

            if isinstance(value, tuple):
                return list(value)

            if isinstance(value, list):
                return value

        return None

    def get_translations(rule):
        for key in (
            "learned_translations",
            "translations",
            "source_translations",
            "translation_rules",
        ):
            value = rule.get(key)

            if isinstance(value, dict):
                return value

        return {}

    def best_target_from_translation(record):
        if record is None:
            return None

        if isinstance(record, str):
            return record

        if isinstance(record, dict):
            if record.get("ambiguous"):
                return None

            for key in (
                "best_target",
                "target",
                "target_key",
                "layout_target",
            ):
                value = record.get(key)

                if value is not None:
                    return value

            counts = record.get("counts")

            if counts is None:
                counts = record.get("target_counts")

            if isinstance(counts, dict) and counts:
                return max(
                    counts,
                    key=lambda item: counts[item],
                )

        return None

    def natural_target_from_fact(fact):
        container_role = fact.get("container_role")
        direction = fact.get("direction")

        if container_role == "outside_all":
            return "outside_right"

        if direction not in ("center", "above", "below", "left", "right"):
            return None

        if container_role == "inner":
            return f"inner_{direction}"

        if container_role == "outer":
            return f"outer_{direction}"

        if container_role == "single":
            return f"outer_{direction}"

        return None

    exact_scene = exact_scene_signature_from_facts(source_facts)
    abstract_scene = abstract_scene_signature_from_facts(source_facts)

    examples = get_examples(scene_layout_rule)

    # ------------------------------------------------------------
    # 1. Exact whole-scene example match.
    # ------------------------------------------------------------
    for example in examples:
        example_exact = get_example_signature(
            example,
            (
                "scene_exact",
                "exact_scene",
                "scene_signature",
                "exact_signature",
            ),
        )

        if example_exact == exact_scene:
            layout = get_example_layout(example)

            if layout is not None:
                return sorted_unique(layout)

    # ------------------------------------------------------------
    # 2. Abstract whole-scene example match.
    # ------------------------------------------------------------
    for example in examples:
        example_abstract = get_example_signature(
            example,
            (
                "scene_abstract",
                "abstract_scene",
                "abstract_signature",
            ),
        )

        if example_abstract == abstract_scene:
            layout = get_example_layout(example)

            if layout is not None:
                return sorted_unique(layout)

    translations = get_translations(scene_layout_rule)

    # ------------------------------------------------------------
    # 3. Per-source learned translations.
    #
    # Use exact translation first.
    # If missing, use abstract translation.
    # If both missing, use natural fallback.
    #
    # This prevents partial learned lists from deleting markers.
    # ------------------------------------------------------------
    targets = []

    for fact in source_facts:
        source_key = fact.get("source_key")
        abstract_key = source_fact_to_abstract_key_local(source_key)

        target = best_target_from_translation(
            translations.get(source_key)
        )

        if target is None:
            target = best_target_from_translation(
                translations.get(abstract_key)
            )

        if target is None:
            target = natural_target_from_fact(fact)

        targets.append(target)

    return sorted_unique(targets)


# ============================================================
# DEBUG PRINTING
# ============================================================

def print_scene_to_layout_rule(scene_layout_rule):
    if scene_layout_rule is None:
        print()
        print("[SCENE TO LAYOUT RULE]")
        print("None")
        return

    print()
    print("[SCENE TO LAYOUT RULE]")
    print(f"family: {scene_layout_rule.get('family')}")
    print(f"mode  : {scene_layout_rule.get('mode')}")

    print("examples:")

    examples = scene_layout_rule.get("examples", [])

    if not examples:
        print("  none")
    else:
        for index, example in enumerate(examples):
            print(f"  example {index}")
            print(f"    scene exact   : {example.get('scene_exact')}")
            print(f"    scene abstract: {example.get('scene_abstract')}")
            print(f"    layout        : {example.get('layout')}")

    print("learned translations:")

    translations = scene_layout_rule.get("learned_translations", {})

    if not translations:
        print("  none")
    else:
        for source_key, mapping in translations.items():
            print(
                f"  {source_key} -> {mapping.get('best_target')} "
                f"counts={mapping.get('target_counts')} "
                f"ambiguous={mapping.get('ambiguous')}"
            )