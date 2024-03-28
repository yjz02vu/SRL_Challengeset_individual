
import json




data = {
  "capabilities": {
    "capability_1": {
      "name": "Voice",
      "tests": {
        "passive": [
          {
            "data": "The play was directed by the playwright.",
            "label": "ARG1",
            "token": "play",
            "predicate": "directed"
          },
          {
            "data": "The project was managed by the supervisor.",
            "label": "ARG1",
            "token": "project",
            "predicate": "managed"
          },
          {
            "data": "The movie was made by the filmmaker.",
            "label": "ARG1",
            "token": "movie",
            "predicate": "made"
          },
          {
            "data": "The painting was created by the artist.",
            "label": "ARG1",
            "token": "painting",
            "predicate": "created"
          },
          {
            "data": "The problem was solved by the scientist.",
            "label": "ARG1",
            "token": "problem",
            "predicate": "solved"
          },
          {
            "data": "The dance performance was choreographed by the dancer.",
            "label": "ARG1",
            "token": "dance",
            "predicate": "choreographed"
          },
          {
            "data": "The building was designed by the architect.",
            "label": "ARG1",
            "token": "building",
            "predicate": "designed"
          },
          {
            "data": "The surgery was performed by the surgeon.",
            "label": "ARG1",
            "token": "surgery",
            "predicate": "performed"
          },
          {
            "data": "The speech was delivered by the politician.",
            "label": "ARG1",
            "token": "speech",
            "predicate": "delivered"
          }
        ]
      },
      "test_type": "MFT",
      "failure_rate": 0,
      "fail": 0
    },
    "capability_2": {
      "name": "Long Distance Dependency",
      "tests": {
        "relative_cause": [
          {
            "data": "The dog, which was adopted last year, loves to play fetch.",
            "label": "ARG0",
            "token": "dog",
            "predicate": "loves"
          },
          {
            "data": "The book, which she borrowed from the library, was a bestseller.",
            "label": "ARG1",
            "token": "book",
            "predicate": "was"
          },
          {
            "data": "The restaurant, where they had their first date, serves delicious Italian cuisine.",
            "label": "ARG0",
            "token": "restaurant",
            "predicate": "serves"
          },
          {
            "data": "Emma, who was one of the richest women in the world, bought an expensive car.",
            "label": "ARG0",
            "token": "car",
            "predicate": "bought"
          },
          {
            "data": "The movie, which won several awards, is now available for streaming.",
            "label": "ARG1",
            "token": "movie",
            "predicate": "is"
          },
          {
            "data": "The house, which was painted blue, stood out in the neighborhood.",
            "label": "ARG1",
            "token": "house",
            "predicate": "stood"
          },
          {
            "data": "The river, which flowed gently, was a peaceful sight to behold.",
            "label": "ARG1",
            "token": "river",
            "predicate": "was"
          },
          {
            "data": "The laptop, which had a long battery life, was ideal for travel.",
            "label": "ARG1",
            "token": "laptop",
            "predicate": "was"
          },
          {
            "data": "The cake, which was decorated with intricate frosting, was shared by Jason's friends.",
            "label": "ARG1",
            "token": "cake",
            "predicate": "shared"
          }
        ]
      },
      "test_type": "MFT",
      "failure_rate": 0,
      "fail": 0
    },
    "capability_3": {
      "name": "Dative Alternation",
      "tests": {
        "test_1": {
          "indirect_object": [
            {
              "data": "Emma gives a present to her sister.",
              "label": "ARG2",
              "token": "sister",
              "predicate": "gives"
            },
            {
              "data": "Luis mailed the postcard to Mindy.",
              "label": "ARG2",
              "token": "Mindy",
              "predicate": "mailed"
            },
            {
              "data": "John gives a book to Wang.",
              "label": "ARG2",
              "token": "Wang",
              "predicate": "gives"
            },
            {
              "data": "Elcotel will forward the billing data to the customer.",
              "label": "ARG2",
              "token": "customer",
              "predicate": "forward"
            },
            {
              "data": "He sends the message to everyone.",
              "label": "ARG2",
              "token": "everyone",
              "predicate": "sends"
            },
            {
              "data": "Steve handed the package to Kim for the mistake.",
              "label": "ARG2",
              "token": "Kim",
              "predicate": "handed"
            },
            {
              "data": "Can you pass the salt to me?",
              "label": "ARG2",
              "token": "me",
              "predicate": "pass"
            },
            {
              "data": "Ziggy is singing a song to her baby.",
              "label": "ARG2",
              "token": "baby",
              "predicate": "singing"
            },
            {
              "data": "Adam introduced the new employee to the team.",
              "label": "ARG2",
              "token": "team",
              "predicate": "introduced"
            }
          ],
          "test_type": "MFT",
          "failure_rate": 0,
          "fail": 0
        },
        "test_2": {
          "double_object": [
            {
              "data": "Emma gives her sister a present.",
              "label": "ARG2",
              "token": "sister",
              "predicate": "gives"
            },
            {
              "data": "Luis mailed Mindy a postcard.",
              "label": "ARG2",
              "token": "Mindy",
              "predicate": "mailed"
            },
            {
              "data": "John gives Wang a book.",
              "label": "ARG2",
              "token": "Wang",
              "predicate": "gives"
            },
            {
              "data": "Kim showed Edward her ring.",
              "label": "ARG2",
              "token": "Edward",
              "predicate": "showed"
            },
            {
              "data": "He sends everyone the message.",
              "label":"ARG2",
              "token": "everyone",
              "predicate": "sends"
            },
            {
              "data": "Steve handed Kim the package for the mistake.",
              "label": "ARG2",
              "token": "Kim",
              "predicate": "handed"
            },
            {
              "data": "Can you pass me the salt?",
              "label": "ARG2",
              "token": "me",
              "predicate": "pass"
            },
            {
              "data": "Ziggy is singing her baby a song.",
              "label": "ARG2",
              "token": "baby",
              "predicate": "singing"
            },
            {
              "data": "Adam throws me his backpack.",
              "label": "ARG2",
              "token": "me",
              "predicate": "throws"
            }
          ],
          "test_type": "MFT",
          "failure_rate": 0,
          "fail": 0
        }
      }
    },
    "capability_4": {
      "name": "Modifiers",
      "tests": {
        "Location_modifier": [
          {
            "data": [
              "Emma works in China.",
              "Emma works in business."
            ],
            "label": [
                "ARGM-LOC",
                "ARG2"
            ],
            "token": [
              "China",
              "business"
            ],
            "predicate": [
              "works",
              "works"
            ]     
          },
          {
            "data": [
              "Danna passed the exam in France. ",
              "Danna passed the exam in finance."
            ],
            "label": [
                "ARGM-LOC",
                "ARG2"
            ],
            "token": [
              "France",
              "finance"
            ],
            "predicate": [
              "passed",
              "passed"
            ]
          },
          {
            "data": [
              "I works in London.",
              "I works in marketing."  
            ], 
            "label": [
              "ARGM-LOC",
              "ARG2"
            ],
            "token": [
              "London",
              "marketing"
            ],
            "predicate": [
              "works",
              "works"
            ]
          },
          {
            "data": [
              "My sister studies in Paris.",
              "My sister studies in law."
            ],
            "label": [
              "ARGM-LOC",
              "ARG2"
            ],
            "token":[
              "Paris" ,
              "law"
            ],
            "predicate": [
              "studies",
              "studies"
            ]
          },
          {
            "data": [
              "The company is based in New York.",
              "The company is based in technology."
            ],
            "label": [
              "ARGM-LOC",
              "ARG2"
            ],
            "token": [
              "New York",
              "technology"
            ],
            "predicate": [
              "based",
              "based"
            ]
          },
          {
            "data": [
              "The school is located in California.",
              "The school is located in science."
            ],
            "label": [
              "ARGM-LOC",
              "ARG2"
            ],
            "token": [
              "California",
              "science"
            ],
            "predicate": [
              "located",
              "located"
            ]
          },
          {
            "data": [
              "My friend works in Hong kong.",
              "My friend works in government."
            ],
            "label": [
              "ARGM-LOC",
              "ARG2"
            ],
            "token": [
              "Hong kong",
              "government"
            ],
            "predicate": [
              "works",
              "works"
            ]
          },
          {
            "data": [
              "He studies in Singapore.",
              "He studies in fashion."
            ],
            "label": [
              "ARGM-LOC",
              "ARG2"
            ],
            "token": [
              "Singapore",
              "fashion"
            ],
            "predicate": [
              "studies",
              "studies"
            ]
          },
          {
            "data": [
              "Mint enjoys her career in Diemen.",
              "Mint enjoys her career in consulting."
            ],
            "label": [
              "ARGM-LOC",
              "ARG2"
            ],
            "token": [
              "Diemen",
              "consulting"
            ],
            "predicate": [
              "enjoys",
              "enjoys"
            ]
          }
        ]
      },
      "test_type": "DIR",
      "failure_rate": 0,
      "fail": 0
    },
    "capability_5": {
      "name": "Robustness",
      "tests": {
        "Negation": [
          {
            "data": [
              "The cat is hungry.",
              "The cat isn't hungry."
            ],
            "label": [
              "ARG1",
              "ARG1"
            ],
            "token": [
              "cat",
              "cat"
            ],
            "predicate": [
              "is",
              "isn't"
            ]
          },
          {
            "data": [
              "The cake was baked by Mary.",
              "The cake wasn't baked by Mary." 
            ],
            "label": [
              "ARG1",
              "ARG1"
            ],
            "token": [
              "cake",
              "cake"
            ],
            "predicate": [
              "baked",
              "baked"
            ]
          },
          {
            "data": [
              "John wrote the report.",
              "John didn't write the report." 
            ],
            "label": [
              "ARG1",
              "ARG1"
            ],
            "token": [
              "report",
              "report"
            ],
            "predicate": [
              "wrote",
              "write"
            ]
          },
          {
            "data": [
              "The presentation was prepared by Emily.",
              "The presentation wasn't prepared by Emily."
            ],
            "label": "ARG1",
            "token": "presentation",
            "predicate": [
              "prepared",
              "prepared"
            ]
          },
          {
            "data": [
              "Alex will take a picture of the house." ,
              "Alex will not take a picture of the house."
            ],
            "label": "ARG1",
            "token": "picture",
            "predicate": [
              "take",
              "take"
            ]
          },
          {
            "data": [
              "The dinner was cooked by Chef Smith." ,
              "The dinner wasn't cooked by Chef Smith."
            ],
            "label": "ARG1",
            "token": "dinner",
            "predicate": [
              "cooked",
              "cooked"
            ]
          },
          {
            "data": [
              "The package was delivered by the postal service." ,
              "The package wasn't delivered by the postal service."
            ],
            "label": "ARG1",
            "token": "package",
            "predicate": [
              "delivered",
              "delivered"
            ]
          },
          {
            "data": [
              "The maintenance team have done the repairs." ,
              "The maintenance team haven't done the repairs."
            ],
            "label": "ARG1",
            "token": "repairs",
            "predicate": [
              "done",
              "done"
            ]
          },
          {
            "data": [
              "The experiment was conducted by the scientists.",
              "The experiment was not conducted by the scientists."
            ],
            "label": "ARG1",
            "token": "experiment",
            "predicate": [
              "conducted",
              "conducted"
            ]
          }
        ]
      },
      "test_type": "MFT",
      "failure_rate": 0,
      "fail": 0
    }
  }
}



json_data = json.dumps(data, indent=2)

# Write JSON data to a file
with open("chanllenge_data.json", "w") as json_file:
    json_file.write(json_data)

print("JSON data has been saved to output.json")