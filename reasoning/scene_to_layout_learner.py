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
    Predict the output marker target set for a new scene.

    Order:
        1. exact whole-scene template match
        2. abstract whole-scene template match
        3. learned source-to-target translations
        4. natural symbolic fallback
    """

    if scene_layout_rule is None:
        return []

    scene_signature = scene_signature_from_source_facts(source_facts)

    exact_key = scene_signature["exact"]
    abstract_key = scene_signature["abstract"]

    exact_templates = scene_layout_rule.get("exact_templates", {})
    abstract_templates = scene_layout_rule.get("abstract_templates", {})
    learned_translations = scene_layout_rule.get("learned_translations", {})

    if exact_key in exact_templates:
        return list(exact_templates[exact_key])

    if abstract_key in abstract_templates:
        return list(abstract_templates[abstract_key])

    targets = []

    for source in source_facts:
        possible_keys = [
            source.get("source_key"),
            source.get("simple_source_key"),
            source_fact_to_abstract_key(source),
        ]

        chosen_target = None

        for key in possible_keys:
            mapping = learned_translations.get(key)

            if mapping is None:
                continue

            if mapping.get("ambiguous"):
                continue

            chosen_target = mapping.get("best_target")
            break

        if chosen_target is None:
            chosen_target = expected_target_for_source(source)

        if chosen_target is not None:
            targets.append(chosen_target)

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