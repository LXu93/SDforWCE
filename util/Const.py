from torchvision import transforms

All_Datasets = ["CAPTIV8", 
               "SEE-AI", 
               "KID2", 
               "Kvasir-Capsule", 
               "WCEBleedGen",
               "SUN",
               "Kvasir v2",
               "Mark-data"]

WCE_Datasets = ["CAPTIV8", 
                "SEE-AI", 
                #"KID2", 
                #"Kvasir-Capsule", 
                #"WCEBleedGen",
                "Mark-data"]

COLN_Datasets = ["Kvasir v2",
                 "SUN"]

Text_Annotation = {"polyp": ["polyp", 
                             "polyps",
                             "Polyp",
                             "polypoids", 
                             "polyp-like",
                             "pseudopolyp"],
                    "non-polyp": ["non-polyp",
                                  "normal",
                                  "Normal clean mucosa"],
                    "inflammation": ["inflammation",
                                     "Inflammation",
                                     "inflammatory change",
                                     "inflammatory",
                                     "erosion",
                                     "Erosion"],
                    "erosion": ["erosion",
                                "Erosiom"],
                    "bleeding": ["bleeding",
                                 "Bleeding",
                                 "blood_fresh",
                                 "blood"],
                    "bubbles":["bubbles"],
                    "dirt": ["dirt"],
                    "clean": ["clean"],
                    "dirt_and_bubbles":["dirt_and_bubbles"],
                    "technical": ["bubbles",
                                  "dirt",
                                  "clean",
                                  "dirt_and_bubbles"],
                    "section": ["stomach",
                                "small intestine",
                                "colon"]
                    
                   }


CROP = {
    "CAPTIV8": (512,512),
    "Mark-data": (512,512),
    "SEE-AI": (512,512),
    "KID2": (320,320),
}

SECTION_OPTS = ["stomach",
                "small intestine",
                "colon",
                "unknown"]

PATHOLOGY_OPTS =["polyp",
                 "non-polyp",
                 "unknown"],
FEATURE_OPTS = ["multiple",
                "single",
                "posible",
                "unknown"]

POSITION_OPTS = ["top-left",
                 "top",
                 "top-right",
                 "left",
                 "center",
                 "right",
                 "bottom-left",
                 "bottom",
                 "bottom-right",
                 "unknown"]

TECH_OPTS = ["bubbles",
             "dirt",
             "clean",
             "unknown"]