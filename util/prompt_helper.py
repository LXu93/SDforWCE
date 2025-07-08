import json
import random
import os
from util import Const

# first version, not used in final
def prompt_generator(pathology, section="unknown", template="An endoscopy image"):
    with open(os.path.join(os.path.abspath(os.getcwd()),'util','prompts.json')) as file:
        prompts_bank = json.load(file)
    if section == "random":
        section = random.choice(Const.Text_Annotation["section"])
    elif section != "unknown":
        template = f"{template} in {section}," 
    if pathology in prompts_bank.keys():
        return f"{template} {random.choice(prompts_bank[pathology])}"
    return template + pathology

def standard_label(label):
    for pathology in Const.Text_Annotation.keys():
        if label in Const.Text_Annotation[pathology]:
            return pathology
    return label

class PromptHelper:
    def __init__(self):
        with open(os.path.join(os.path.abspath(os.getcwd()),'util','template_bank.json')) as f:
            self.template_bank = json.load(f)

    def random_vanish(self, attribute, p=0.5):
        if attribute != "unknown":
            if random.random() < p:
                return "unknown"
            else:
                return attribute
        else: return "unknown"

    def get_template_key(self, section, pathology, feature, position, tech, drop=True):
        attributes = {}
        if drop:
            section = self.random_vanish(section, 0.1)
            feature = self.random_vanish(feature,0.3)
            position = self.random_vanish(position,0.3)
            tech = self.random_vanish(tech, 0.4)
        if section and section != "unknown":
            attributes["section"] = section
        if pathology and pathology != "unknown":
            if pathology.startswith("non"):
                attributes["non_pathology"] = pathology[4:]
            else:
                attributes["pathology"] = pathology
        if feature and feature != "unknown":
            attributes["feature"] = feature
        if position and position != "unknown":
            attributes['position'] = position
        if tech and tech != "unknown":
            if tech == "clean":
                attributes["tech"] = random.choice(["clean view",
                                                    "no bubble",
                                                    "no dirt"])
            else:
                attributes["tech"] = tech
        return attributes

    def generate_prompt(self, section="unknown", pathology="unknown", feature="unknown", position="unknown", tech="unknown"):
        attributes = self.get_template_key(section, pathology, feature, position, tech, drop=True)
        keys = set(attributes.keys()) # use set to ignore order
        template = None
        for k in self.template_bank:
            if set(k.split('|')) == keys:
                template = random.choice(self.template_bank[k])
                break

        if template is None:
            raise ValueError(f"No matching template for attribute keys: {keys}")

        prompt = template.format(**attributes)

        return prompt
    
    def generate_prompt_infer(self, **kwargs):
        if "section" in kwargs.keys():
            section = kwargs["section"]
        else: section = random.choices(Const.SECTION_OPTS, weights=[0.2, 0.7, 0.1, 0.0], k=1)[0]
        if "pathology" in kwargs.keys():
            pathology = kwargs["pathology"]
        else: pathology = random.choice(Const.PATHOLOGY_OPTS)
        if "feature" in kwargs.keys():
            feature = kwargs["feature"]
        else: feature = random.choice(Const.FEATURE_OPTS)
        if "position" in kwargs.keys():
            position = kwargs["position"]
        else: position = random.choice(Const.POSITION_OPTS)
        if "tech" in kwargs.keys():
            tech = kwargs["tech"]
        else: tech = random.choice(Const.TECH_OPTS)
        attributes = self.get_template_key(section, pathology, feature, position, tech, drop=False)
        keys = set(attributes.keys())
        template = None
        for k in self.template_bank:
            if set(k.split('|')) == keys:
                template = random.choice(self.template_bank[k])
                break

        if template is None:
            raise ValueError(f"No matching template for attribute keys: {keys}")

        prompt = template.format(**attributes)
        return prompt
    
if __name__ == "__main__":
    prompt_generator = PromptHelper()
    print(prompt_generator.generate_prompt(section="colon",pathology="non-polyp" ))
