"""Candidate lexicons for the in-context load experiment.

All lists are *candidates*: they are filtered at runtime to entries that
tokenize to a single token with a leading space in the target tokenizer
(see items.filter_single_token). Pools must survive filtering with at
least N_max entries; items.build_pools raises otherwise.

Entity classes set the key-similarity axis rho:
  high  -- one narrow lexical class (first names)
  mid   -- natural kinds (animals + plants + foods)
  low   -- maximal mixture across disjoint categories
Values are common surnames, disjoint from every entity pool.
rho is verified by measured Delta, never assumed (write-up, Setup).
"""

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher",
    "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret",
    "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Paul",
    "Emily", "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Dorothy",
    "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa", "Edward",
    "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary",
    "Amy", "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna",
    "Stephen", "Brenda", "Larry", "Pamela", "Justin", "Emma", "Scott",
    "Nicole", "Brandon", "Helen", "Benjamin", "Samantha", "Samuel",
    "Katherine", "Gregory", "Christine", "Frank", "Debra", "Alexander",
    "Rachel", "Raymond", "Carolyn", "Patrick", "Janet", "Jack", "Catherine",
    "Dennis", "Maria", "Jerry", "Heather", "Tyler", "Diane", "Aaron", "Ruth",
    "Jose", "Julie", "Adam", "Olivia", "Nathan", "Joyce", "Henry", "Virginia",
    "Douglas", "Victoria", "Zachary", "Kelly", "Peter", "Lauren", "Kyle",
    "Christina", "Ethan", "Joan", "Walter", "Evelyn", "Noah", "Judith",
    "Jeremy", "Megan", "Christian", "Andrea", "Keith", "Cheryl", "Roger",
    "Hannah", "Terry", "Jacqueline", "Austin", "Martha", "Sean", "Gloria",
    "Gerald", "Teresa", "Carl", "Ann", "Harold", "Sara", "Dylan", "Madison",
    "Arthur", "Frances", "Lawrence", "Kathryn", "Jordan", "Janice", "Jesse",
    "Jean", "Bryan", "Abigail", "Billy", "Alice", "Bruce", "Julia", "Gabriel",
    "Judy", "Joe", "Sophia", "Logan", "Grace", "Alan", "Denise", "Juan",
    "Amber", "Albert", "Doris", "Willie", "Marilyn", "Elijah", "Danielle",
    "Wayne", "Beverly", "Randy", "Isabella", "Vincent", "Theresa", "Mason",
    "Diana", "Roy", "Natalie", "Ralph", "Brittany", "Bobby", "Charlotte",
    "Russell", "Marie", "Bradley", "Kayla", "Philip", "Alexis", "Eugene",
    "Lori", "Louis", "Tiffany", "Harry", "Kathy", "Wade", "Blake", "Colin",
    "Bernard", "Leroy", "Marcus", "Theodore", "Clifford", "Miguel", "Oscar",
    "Jay", "Jim", "Tom", "Calvin", "Alex", "Jon", "Ronnie", "Bill", "Lloyd",
    "Tommy", "Leon", "Derek", "Warren", "Darrell", "Jerome", "Floyd", "Leo",
    "Alvin", "Tim", "Wesley", "Gordon", "Dean", "Greg", "Jorge", "Dustin",
    "Pedro", "Derrick", "Dan", "Lewis", "Zachariah", "Corey", "Herman",
    "Maurice", "Vernon", "Roberto", "Clyde", "Glen", "Hector", "Shane",
    "Ricardo", "Sam", "Rick", "Lester", "Brent", "Ramon", "Charlie", "Tim",
    "Elmer", "Brad", "Gabe", "Ron", "Mitchell", "Roland", "Arnold", "Harvey",
    "Jared", "Adrian", "Karl", "Cory", "Claude", "Erik", "Darryl", "Jamie",
    "Neil", "Jessie", "Christina", "Rose", "Anne", "Jane", "Lucy", "Ellen",
    "Clara", "Molly", "Nora", "Stella", "Vera", "Iris", "Ivy", "Faith",
    "Hope", "Joy", "June", "April", "May", "Dawn", "Eve", "Gail", "Kate",
    "Kim", "Lynn", "Beth", "Jill", "Joanna", "Kristen", "Erica", "Monica",
    "Tracy", "Wendy", "Vanessa", "Priscilla", "Sylvia", "Rosa", "Irene",
    "Ana", "Alma", "Vivian", "Leah", "Naomi", "Ada", "Edith", "Elsie",
    "Ethel", "Hazel", "Mabel", "Pearl", "Ruby", "Sadie", "Tina", "Gina",
    "Nina", "Rita", "Sonia", "Tanya", "Paula", "Carla", "Marta", "Elena",
    "Sofia", "Lydia", "Miriam", "Esther", "Bonnie", "Connie", "Debbie",
    "Peggy", "Sally", "Sue", "Terri", "Toni", "Vicki", "Annie", "Carrie",
    "Cassie", "Daisy", "Flora", "Greta", "Hattie", "Josie", "Lena", "Lila",
    "Lola", "Mae", "Nell", "Opal", "Reba", "Rosie", "Vada", "Willa",
]

SURNAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Taylor", "Moore", "Jackson", "Martin", "Lee",
    "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez",
    "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams",
    "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter",
    "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales",
    "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper",
    "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox",
    "Ward", "Richardson", "Watson", "Brooks", "Chavez", "Wood", "James",
    "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes", "Price", "Alvarez",
    "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross", "Foster",
    "Jimenez", "Powell", "Jenkins", "Perry", "Russell", "Sullivan", "Bell",
    "Coleman", "Butler", "Henderson", "Barnes", "Gonzales", "Fisher",
    "Vasquez", "Simmons", "Romero", "Jordan", "Patterson", "Alexander",
    "Hamilton", "Graham", "Reynolds", "Griffin", "Wallace", "Moreno",
    "West", "Cole", "Hayes", "Bryant", "Herrera", "Gibson", "Ellis",
    "Tran", "Medina", "Aguilar", "Stevens", "Murray", "Ford", "Castro",
    "Marshall", "Owens", "Harrison", "Fernandez", "McDonald", "Woods",
    "Washington", "Kennedy", "Wells", "Vargas", "Henry", "Chen", "Freeman",
    "Webb", "Tucker", "Guzman", "Burns", "Crawford", "Olson", "Simpson",
    "Porter", "Hunter", "Gordon", "Mendez", "Silva", "Shaw", "Snyder",
    "Mason", "Dixon", "Munoz", "Hunt", "Hicks", "Holmes", "Palmer", "Wagner",
    "Black", "Robertson", "Boyd", "Rose", "Stone", "Salazar", "Fox",
    "Warren", "Mills", "Meyer", "Rice", "Schmidt", "Garza", "Daniels",
    "Ferguson", "Nichols", "Stephens", "Soto", "Weaver", "Ryan", "Gardner",
    "Payne", "Grant", "Dunn", "Kelley", "Spencer", "Hawkins", "Arnold",
    "Pierce", "Vazquez", "Hansen", "Peters", "Santos", "Hart", "Bradley",
    "Knight", "Elliott", "Cunningham", "Duncan", "Armstrong", "Hudson",
    "Carroll", "Lane", "Riley", "Andrews", "Alvarado", "Ray", "Delgado",
    "Berry", "Perkins", "Hoffman", "Johnston", "Matthews", "Pena",
    "Richards", "Contreras", "Willis", "Carpenter", "Lawrence", "Sandoval",
    "Guerrero", "George", "Chapman", "Rios", "Estrada", "Ortega", "Watkins",
    "Greene", "Nunez", "Wheeler", "Valdez", "Harper", "Burke", "Larson",
    "Santiago", "Maldonado", "Morrison", "Franklin", "Carlson", "Austin",
    "Dominguez", "Carr", "Lawson", "Jacobs", "Obrien", "Lynch", "Singh",
    "Vega", "Bishop", "Montgomery", "Oliver", "Jensen", "Harvey", "Williamson",
    "Gilbert", "Dean", "Sims", "Espinoza", "Howell", "Li", "Wong", "Reid",
    "Hanson", "Le", "McCoy", "Garrett", "Burton", "Fuller", "Wang", "Weber",
    "Welch", "Rojas", "Lucas", "Marquez", "Fields", "Park", "Yang", "Little",
    "Banks", "Padilla", "Day", "Walsh", "Bowman", "Schultz", "Luna", "Fowler",
    "Mejia", "Davidson", "Acosta", "Brewer", "May", "Holland", "Juarez",
    "Newman", "Pearson", "Curtis", "Cortez", "Douglas", "Schneider", "Joseph",
    "Barrett", "Navarro", "Figueroa", "Keller", "Avila", "Wade", "Molina",
    "Stanley", "Hopkins", "Campos", "Barnett", "Bates", "Chambers", "Caldwell",
    "Beck", "Lambert", "Miranda", "Byrd", "Craig", "Ayala", "Lowe", "Frazier",
    "Powers", "Neal", "Leonard", "Gregory", "Carrillo", "Sutton", "Fleming",
]

ANIMALS = [
    "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken", "duck",
    "goose", "rabbit", "mouse", "rat", "bear", "wolf", "fox", "deer", "moose",
    "elk", "lion", "tiger", "leopard", "cheetah", "zebra", "giraffe",
    "elephant", "rhino", "hippo", "monkey", "ape", "gorilla", "camel",
    "donkey", "mule", "bull", "ox", "bison", "buffalo", "seal", "whale",
    "dolphin", "shark", "fish", "salmon", "trout", "bass", "cod", "tuna",
    "eel", "crab", "lobster", "shrimp", "squid", "octopus", "snail", "worm",
    "spider", "ant", "bee", "wasp", "fly", "moth", "beetle", "cricket",
    "snake", "lizard", "turtle", "frog", "toad", "owl", "hawk", "eagle",
    "falcon", "crow", "raven", "robin", "sparrow", "finch", "pigeon", "dove",
    "swan", "heron", "crane", "stork", "penguin", "parrot", "peacock",
    "turkey", "bat", "otter", "beaver", "badger", "skunk", "raccoon",
    "squirrel", "hedgehog", "mole", "ferret", "weasel", "lynx", "cougar",
    "panther", "jaguar", "hyena", "jackal", "koala", "kangaroo", "panda",
    "sloth", "lemur", "gecko", "iguana", "cobra", "viper", "python",
]

PLANTS_FOODS = [
    "apple", "pear", "peach", "plum", "cherry", "grape", "orange", "lemon",
    "lime", "banana", "mango", "melon", "berry", "fig", "date", "olive",
    "corn", "wheat", "rice", "oat", "barley", "rye", "bean", "pea", "lentil",
    "potato", "carrot", "onion", "garlic", "pepper", "tomato", "cabbage",
    "lettuce", "spinach", "celery", "radish", "beet", "turnip", "squash",
    "pumpkin", "cucumber", "eggplant", "broccoli", "mushroom", "ginger",
    "mint", "basil", "thyme", "sage", "parsley", "pine", "oak", "maple",
    "birch", "cedar", "willow", "elm", "ash", "fir", "spruce", "palm",
    "bamboo", "fern", "moss", "ivy", "rose", "tulip", "daisy", "lily",
    "orchid", "violet", "poppy", "sunflower", "clover", "cactus", "bread",
    "cheese", "butter", "cream", "milk", "honey", "sugar", "salt", "flour",
    "egg", "meat", "beef", "pork", "lamb", "bacon", "ham", "sausage",
    "soup", "stew", "salad", "pasta", "noodle", "sauce", "vinegar", "mustard",
    "pickle", "jam", "syrup", "cake", "pie", "cookie", "candy", "chocolate",
    "coffee", "tea", "juice", "cider", "wine", "beer", "walnut", "almond",
    "peanut", "cashew", "coconut", "raisin", "prune", "apricot",
]

OBJECTS_TOOLS = [
    "hammer", "wrench", "saw", "drill", "nail", "screw", "bolt", "chain",
    "rope", "wire", "pipe", "valve", "pump", "gear", "spring", "lever",
    "wheel", "axle", "engine", "motor", "battery", "switch", "lamp", "bulb",
    "candle", "torch", "clock", "watch", "compass", "scale", "ruler", "knife",
    "fork", "spoon", "plate", "bowl", "cup", "mug", "jar", "bottle", "bucket",
    "basket", "box", "crate", "barrel", "chest", "drawer", "shelf", "table",
    "chair", "bench", "stool", "couch", "bed", "pillow", "blanket", "curtain",
    "mirror", "frame", "door", "window", "ladder", "stairs", "fence", "gate",
    "brush", "broom", "mop", "sponge", "towel", "soap", "razor", "comb",
    "needle", "thread", "scissors", "glue", "tape", "pen", "pencil", "paper",
    "book", "map", "card", "coin", "key", "lock", "bell", "whistle", "drum",
    "flute", "violin", "piano", "guitar", "trumpet", "harp", "anchor", "oar",
    "sail", "mast", "net", "hook", "spear", "shield", "sword", "arrow", "bow",
]

PLACES = [
    "London", "Paris", "Berlin", "Madrid", "Rome", "Vienna", "Prague",
    "Moscow", "Athens", "Dublin", "Oslo", "Stockholm", "Helsinki", "Warsaw",
    "Lisbon", "Brussels", "Amsterdam", "Geneva", "Zurich", "Munich",
    "Hamburg", "Milan", "Naples", "Venice", "Florence", "Barcelona",
    "Seville", "Porto", "Cairo", "Nairobi", "Lagos", "Tokyo", "Kyoto",
    "Osaka", "Seoul", "Beijing", "Shanghai", "Delhi", "Mumbai", "Bangkok",
    "Singapore", "Manila", "Jakarta", "Sydney", "Melbourne", "Auckland",
    "Toronto", "Montreal", "Vancouver", "Boston", "Chicago", "Denver",
    "Houston", "Dallas", "Austin", "Seattle", "Portland", "Phoenix",
    "Atlanta", "Miami", "Orlando", "Detroit", "Cleveland", "Memphis",
    "Nashville", "Baltimore", "Pittsburgh", "Cincinnati", "Milwaukee",
    "Omaha", "Tulsa", "Wichita", "Tucson", "Fresno", "Oakland", "Sacramento",
]

PROFESSIONS = [
    "doctor", "nurse", "teacher", "lawyer", "judge", "farmer", "baker",
    "butcher", "carpenter", "plumber", "electrician", "mechanic", "engineer",
    "architect", "painter", "sculptor", "writer", "poet", "singer", "dancer",
    "actor", "chef", "waiter", "barber", "tailor", "cobbler", "miner",
    "sailor", "pilot", "driver", "soldier", "officer", "detective", "guard",
    "banker", "merchant", "trader", "clerk", "cashier", "librarian",
    "professor", "student", "scientist", "chemist", "biologist", "physicist",
    "astronomer", "geologist", "historian", "translator", "editor",
    "reporter", "photographer", "designer", "programmer", "analyst",
    "accountant", "auditor", "surgeon", "dentist", "pharmacist", "therapist",
    "veterinarian", "shepherd", "fisherman", "hunter", "blacksmith",
    "goldsmith", "jeweler", "potter", "weaver", "gardener", "janitor",
]

NATURE_WEATHER = [
    "rain", "snow", "hail", "sleet", "fog", "mist", "cloud", "storm",
    "thunder", "lightning", "wind", "breeze", "gale", "frost", "ice", "dew",
    "sun", "moon", "star", "comet", "meteor", "planet", "sky", "dawn",
    "dusk", "sunset", "sunrise", "river", "stream", "creek", "lake", "pond",
    "ocean", "sea", "bay", "gulf", "tide", "wave", "beach", "shore", "cliff",
    "mountain", "hill", "valley", "canyon", "plain", "desert", "forest",
    "jungle", "swamp", "marsh", "meadow", "field", "prairie", "tundra",
    "glacier", "volcano", "island", "peninsula", "cave", "spring", "summer",
    "autumn", "winter", "sand", "clay", "mud", "dust", "gravel", "boulder",
    "pebble", "quartz", "granite", "marble", "slate", "coal", "iron",
    "copper", "silver", "gold", "lead", "tin", "zinc", "steel", "bronze",
    "brass", "amber", "pearl", "ruby", "emerald", "diamond", "jade", "opal",
]

VEHICLES_CLOTHING = [
    "car", "truck", "bus", "van", "jeep", "taxi", "train", "tram", "subway",
    "bicycle", "motorcycle", "scooter", "boat", "ship", "ferry", "canoe",
    "kayak", "yacht", "raft", "barge", "plane", "jet", "helicopter",
    "glider", "balloon", "rocket", "sled", "wagon", "cart", "carriage",
    "tractor", "crane", "shirt", "pants", "dress", "skirt", "coat",
    "jacket", "sweater", "vest", "suit", "tie", "scarf", "hat", "cap",
    "helmet", "glove", "mitten", "sock", "shoe", "boot", "sandal",
    "slipper", "belt", "buckle", "button", "zipper", "pocket", "collar",
    "sleeve", "apron", "robe", "gown", "uniform", "badge", "ribbon",
    "crown", "ring", "necklace", "bracelet", "earring", "brooch", "veil",
]

ENTITY_CLASSES = {
    "high": {"names": FIRST_NAMES},
    "mid": {"animals": ANIMALS, "plants_foods": PLANTS_FOODS,
            "objects_tools": OBJECTS_TOOLS},
    "low": {
        "animals": ANIMALS, "plants_foods": PLANTS_FOODS,
        "objects_tools": OBJECTS_TOOLS, "places": PLACES,
        "professions": PROFESSIONS, "nature_weather": NATURE_WEATHER,
        "vehicles_clothing": VEHICLES_CLOTHING,
    },
}

VALUE_POOL = SURNAMES

# Original neutral prose written for this project (not sampled from a corpus,
# so it cannot be memorized verbatim by the model). Deliberately abstract and
# procedural to minimize lexical collision with the entity and value pools;
# items.py additionally drops any filler chunk whose tokens collide with the
# sampled items of a given prompt.
FILLER_TEXT = (
    "The committee reviewed the proposal in three separate sessions before "
    "reaching a decision. Each session began with a summary of the previous "
    "discussion and ended with a list of unresolved questions. The first "
    "session focused on the scope of the plan, the second on its schedule, "
    "and the third on the division of responsibilities among the members. "
    "Minutes were recorded throughout and circulated for correction within "
    "a week. Several amendments were suggested during the review period, "
    "most of which concerned the wording of the second section rather than "
    "its substance. The revised draft was shorter than the original and "
    "arranged its arguments in a different order. A final vote was scheduled "
    "for the following month, pending the outcome of two smaller inquiries. "
    "The procedure for submitting comments was described in an appendix, "
    "which explained the format required and the deadline for each stage. "
    "Responses arrived steadily at first and then more slowly as the "
    "deadline approached. A small working group was appointed to sort the "
    "responses into categories and to prepare a short report on the general "
    "pattern of opinion. The report noted broad agreement on the overall "
    "aim and considerable disagreement about the means of achieving it. "
    "Some responses proposed delaying the entire process until further "
    "information became available, while others argued that any delay "
    "would only increase the eventual cost. The working group declined to "
    "take a position on this question and confined itself to describing "
    "the range of views received. Its report was accepted without "
    "amendment and attached to the main file for future reference. The "
    "next phase of the process required each participant to state in "
    "writing which of the proposed options they preferred and why. These "
    "statements varied greatly in length and detail, from a single "
    "sentence to several pages of argument. A summary table was prepared "
    "showing the distribution of preferences across the options, together "
    "with a note on how the totals should be interpreted. The table showed "
    "no clear majority for any single option, though two of them together "
    "accounted for most of the stated preferences. Further consultation "
    "was therefore recommended before any binding choice was made. The "
    "consultation took the form of a series of open meetings, each devoted "
    "to one aspect of the question and each concluding with a period for "
    "general remarks. Attendance was highest at the first meeting and "
    "declined gradually thereafter, a pattern the organizers had expected. "
    "Notes from every meeting were combined into a single document, "
    "indexed by topic, and made available on request. When the schedule "
    "of meetings ended, a drafting group assembled the accumulated "
    "material into a consolidated statement of the issues. The statement "
    "was organized into numbered paragraphs so that later references "
    "could be precise. Paragraphs describing points of agreement were "
    "placed first, followed by those describing points of dispute, each "
    "with a brief account of the main arguments on either side. The "
    "drafting group took care to phrase the disputed points neutrally, "
    "and the final text was checked by two independent readers for "
    "balance. Publication followed in the usual manner, and a period of "
    "six weeks was allowed for formal objections. Only a handful of "
    "objections were received, and none raised a point that had not "
    "already been considered in the earlier stages of the process."
)

FILLER_TEXT_2 = (
    "A separate question concerned the manner in which the results of the "
    "process should be recorded for those who might consult them later. "
    "Opinion divided between a short account limited to the conclusions "
    "and a longer account preserving the reasoning behind each step. The "
    "longer form was chosen on the ground that conclusions detached from "
    "their reasons are difficult to apply to new circumstances. An "
    "editorial group was given the task of preparing this account, with "
    "instructions to favor plain wording over technical terms wherever "
    "the meaning allowed it. Drafts circulated twice before the wording "
    "settled, and the second circulation produced far fewer comments "
    "than the first. The completed account ran to forty pages, arranged "
    "chronologically, with an index of decisions at the end. Copies were "
    "deposited in the usual places and a notice of their availability "
    "was published in the ordinary way. Attention then turned to the "
    "question of implementation, which had deliberately been set aside "
    "until the record was complete. A schedule of stages was drawn up, "
    "each stage ending with a review at which progress would be compared "
    "with the original intention. Provision was made for adjusting the "
    "schedule if any review found a substantial gap between the two. "
    "Responsibility for each stage was assigned to a named group, and "
    "the groups were asked to report in a common format so that their "
    "accounts could be compared directly. The first review found the "
    "work slightly ahead of expectation, the second found it slightly "
    "behind, and neither difference was judged large enough to justify "
    "a change of plan. At the close of the final stage a general meeting "
    "was held to consider whether the aims stated at the outset had been "
    "achieved. The prevailing view was that most of them had, that some "
    "had been overtaken by events, and that one or two had proved less "
    "important than they had seemed at the beginning. A short closing "
    "statement to this effect was adopted, thanks were recorded to all "
    "who had taken part, and the file was formally closed subject to "
    "reopening if circumstances required it at any future time."
)

FILLER_TEXT_3 = (
    "Looking back on the whole sequence of events, several of those "
    "involved offered observations about what had gone well and what "
    "might have been arranged differently. It was generally felt that "
    "the early stages had taken longer than necessary, chiefly because "
    "the purpose of each step had not been stated clearly enough at the "
    "start. Against this, the later stages had moved quickly, and the "
    "habit of recording decisions as they were made had saved a great "
    "deal of effort near the end. One participant remarked that the "
    "most useful single practice had been the standing rule that every "
    "objection must be accompanied by a stated alternative, which had "
    "kept the discussion moving even when opinions were far apart. "
    "Another pointed to the value of circulating drafts well before "
    "each meeting rather than presenting them at the meeting itself. "
    "These observations were gathered into a brief note of guidance "
    "for any group undertaking a similar exercise in the future. The "
    "note made no claim to completeness and was offered simply as a "
    "record of experience, to be weighed against the circumstances of "
    "whatever occasion might later arise and revised freely as those "
    "circumstances required, without any obligation to the original."
)

FILLER_TEXT = FILLER_TEXT + " " + FILLER_TEXT_2 + " " + FILLER_TEXT_3
