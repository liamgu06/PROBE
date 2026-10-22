import random
import json


vocab = {
    "attribute": [
        "action", "activity", "behavior", "event", "event structure",
        "interaction", "interaction pattern", "social interaction", "motion",
        "movement pattern", "motion dynamics", "emotional expression",
        "affective tone", "emotional intensity", "intention",
        "goal-directed behavior", "communication pattern", "engagement level",
        "attention pattern", "reaction", "response pattern", "temporal structure",
        "temporal organization", "narrative structure", "story progression",
        "causal relationship", "cause-effect pattern", "decision-making process",
        "transition between events", "overall situation"
    ],

    "context": [
        "the scene context", "the surrounding environment", "the situational context",
        "the physical setting", "the social context", "the overall setting",
        "the background conditions", "the situational backdrop"
    ],

    "expression_mode": [
        "expressed", "demonstrated", "manifested", "revealed", "conveyed",
        "displayed", "organized", "structured", "signaled", "reflected"
    ],

    "temporal_aspect": [
        "temporal progression", "sequence of events", "ordering over time",
        "changes over time", "timing and duration", "temporal consistency",
        "temporal transitions", "the overall timeline"
    ],

    "relational_aspect": [
        "relationships between events", "dependencies between actions",
        "connections among moments", "cause-and-effect relations",
        "logical connections", "structural relationships"
    ],

    "dynamic_property": [
        "variation", "stability", "adaptation", "transition",
        "escalation", "de-escalation", "accumulation",
        "repetition", "rhythmic pattern"
    ],

    "interaction_pattern": [
        "coordination", "turn-taking", "mutual influence",
        "reciprocal behavior", "collective behavior",
        "interaction flow", "interaction dynamics"
    ],

    "abstraction_level": [
        "a concrete level", "a mid-level interpretation",
        "a high-level interpretation", "an abstract level",
        "a conceptual level", "a semantic level"
    ],

    "focus_aspect": [
        "overall structure", "high-level patterns", "how it unfolds",
        "relationships between events", "the way it is organized",
        "the underlying structure", "the semantic pattern"
    ],

    "ignored_aspect": [
        "specific visual details", "appearance differences",
        "stylistic variation", "background noise",
        "low-level visual features"
    ]
}


skeletons = {
    "Skeleton 1": {
        "desc": "attribute × expression × context",
        "instruction": "Identify the {attribute} in the video and describe how it is {expression_mode}, including {context}.",
        "task": "Describe the {attribute}, including how it unfolds, how it is structured, and how it relates to the overall situation."
    },
    "Skeleton 2": {
        "desc": "attribute × temporal × relational",
        "instruction": "Identify the {attribute} in the video and describe its {temporal_aspect} and {relational_aspect}.",
        "task": "Describe the {attribute}, focusing on how events are connected over time and how different elements are related."
    },
    "Skeleton 3": {
        "desc": "attribute × abstraction × focus",
        "instruction": "Identify the {attribute} conveyed by the video and describe it at {abstraction_level}.",
        "task": "Describe the {attribute}, emphasizing {focus_aspect} rather than surface-level visual details."
    },
    "Skeleton 4": {
        "desc": "attribute × dynamics × environment",
        "instruction": "Identify the {attribute} in the video and describe its {dynamic_property} in relation to {context}.",
        "task": "Describe the {attribute}, including how it changes, adapts, or responds within the environment."
    },
    "Skeleton 5": {
        "desc": "attribute × interaction-structure",
        "instruction": "Identify the {attribute} involving interaction in the video and describe its {interaction_pattern}.",
        "task": "Describe the {attribute}, focusing on how interactions are organized and how they evolve within the scene."
    },
    "Skeleton 6": {
        "desc": "attribute × constraint / control",
        "instruction": "Identify the {attribute} in the video, focusing on {focus_aspect} while ignoring {ignored_aspect}.",
        "task": "Describe the {attribute} under this constraint and explain how it is manifested."
    }
}


def generate_prompts(num_per_skeleton=5):

    results = []

    for sk_name, sk_data in skeletons.items():
        for _ in range(num_per_skeleton):
            current_attribute = random.choice(vocab["attribute"])
            
            fill_dict = {
                "attribute": current_attribute,
                "context": random.choice(vocab["context"]),
                "expression_mode": random.choice(vocab["expression_mode"]),
                "temporal_aspect": random.choice(vocab["temporal_aspect"]),
                "relational_aspect": random.choice(vocab["relational_aspect"]),
                "dynamic_property": random.choice(vocab["dynamic_property"]),
                "interaction_pattern": random.choice(vocab["interaction_pattern"]),
                "abstraction_level": random.choice(vocab["abstraction_level"]),
                "focus_aspect": random.choice(vocab["focus_aspect"]),
                "ignored_aspect": random.choice(vocab["ignored_aspect"])
            }

            try:
                inst = sk_data["instruction"].format(**fill_dict)
                task = sk_data["task"].format(**fill_dict)
                
                results.append({
                    "skeleton_id": sk_name,
                    "description": sk_data["desc"],
                    "prompt": f"{inst}\n{task}"
                })
            except KeyError as e:
                print(f"Error generating {sk_name}: Missing key {e}")

    return results

if __name__ == "__main__":
    generated_data = generate_prompts(num_per_skeleton=700)

    print(f"Total Prompts Generated: {len(generated_data)}\n")
    
    for idx, item in enumerate(generated_data, 1):
        print(f"--- Sample {idx} [{item['skeleton_id']}] ---")
        
        print("")

    with open('generated_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=4)