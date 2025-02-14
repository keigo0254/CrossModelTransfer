import os

import numpy as np
import torch
import torchvision

from .common import ImageFolderWithPaths, SubsetSampler


imagenet_classnames = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",        # noqa: E501
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",          # noqa: E501
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",      # noqa: E501
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",                # noqa: E501
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",           # noqa: E501
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",         # noqa: E501
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",           # noqa: E501
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",            # noqa: E501
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",        # noqa: E501
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",              # noqa: E501
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",              # noqa: E501
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",            # noqa: E501
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",        # noqa: E501
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",     # noqa: E501
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",       # noqa: E501
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",                      # noqa: E501
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",        # noqa: E501
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",      # noqa: E501
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",         # noqa: E501
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",    # noqa: E501
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",               # noqa: E501
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",        # noqa: E501
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",            # noqa: E501
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",         # noqa: E501
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",           # noqa: E501
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",         # noqa: E501
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",          # noqa: E501
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",          # noqa: E501
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",      # noqa: E501
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",                       # noqa: E501
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",                      # noqa: E501
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",                      # noqa: E501
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",                        # noqa: E501
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",            # noqa: E501
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",                     # noqa: E501
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",           # noqa: E501
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",                # noqa: E501
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",           # noqa: E501
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",               # noqa: E501
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",           # noqa: E501
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",             # noqa: E501
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",             # noqa: E501
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",      # noqa: E501
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",      # noqa: E501
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",       # noqa: E501
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",                   # noqa: E501
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",         # noqa: E501
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",                   # noqa: E501
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",     # noqa: E501
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",       # noqa: E501
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",        # noqa: E501
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",             # noqa: E501
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",             # noqa: E501
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",            # noqa: E501
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",                   # noqa: E501
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",            # noqa: E501
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",                 # noqa: E501
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",     # noqa: E501
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",             # noqa: E501
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",    # noqa: E501
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",             # noqa: E501
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",        # noqa: E501
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",    # noqa: E501
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",        # noqa: E501
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",       # noqa: E501
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",               # noqa: E501
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",               # noqa: E501
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",               # noqa: E501
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",               # noqa: E501
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",    # noqa: E501
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",              # noqa: E501
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",              # noqa: E501
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",      # noqa: E501
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",       # noqa: E501
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",             # noqa: E501
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",                  # noqa: E501
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",      # noqa: E501
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",    # noqa: E501
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",                       # noqa: E501
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",             # noqa: E501
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",           # noqa: E501
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",      # noqa: E501
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",        # noqa: E501
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",         # noqa: E501
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",        # noqa: E501
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",             # noqa: E501
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",               # noqa: E501
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",                # noqa: E501
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",                # noqa: E501
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",                # noqa: E501
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",     # noqa: E501
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",                   # noqa: E501
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",       # noqa: E501
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",      # noqa: E501
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",      # noqa: E501
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",     # noqa: E501
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",      # noqa: E501
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",    # noqa: E501
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",             # noqa: E501
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",           # noqa: E501
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",      # noqa: E501
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",       # noqa: E501
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",      # noqa: E501
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",            # noqa: E501
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",          # noqa: E501
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",        # noqa: E501
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",     # noqa: E501
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",     # noqa: E501
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",            # noqa: E501
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",                  # noqa: E501
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",       # noqa: E501
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",       # noqa: E501
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",                # noqa: E501
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",     # noqa: E501
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",        # noqa: E501
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",                    # noqa: E501
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",              # noqa: E501
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",    # noqa: E501
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",            # noqa: E501
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",            # noqa: E501
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",            # noqa: E501
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",       # noqa: E501
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",               # noqa: E501
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",                    # noqa: E501
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",       # noqa: E501
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",    # noqa: E501
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",        # noqa: E501
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",       # noqa: E501
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",          # noqa: E501
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",        # noqa: E501
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",       # noqa: E501
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",           # noqa: E501
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",        # noqa: E501
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",           # noqa: E501
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",              # noqa: E501
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",       # noqa: E501
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",               # noqa: E501
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",           # noqa: E501
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",            # noqa: E501
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",          # noqa: E501
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",           # noqa: E501
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",          # noqa: E501
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",     # noqa: E501
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",                  # noqa: E501
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",                     # noqa: E501
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",            # noqa: E501
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",         # noqa: E501
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",      # noqa: E501
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",        # noqa: E501
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",         # noqa: E501
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",            # noqa: E501
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",      # noqa: E501
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",    # noqa: E501
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",     # noqa: E501
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",          # noqa: E501
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",           # noqa: E501
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",    # noqa: E501
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",         # noqa: E501
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",     # noqa: E501
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"              # noqa: E501
]


class ImageNet:
    """ImageNetデータセットのラッパークラス"""
    def __init__(
        self,
        preprocess: torchvision.transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        """
        ImageNetデータセットのラッパークラスを初期化

        Args:
            preprocess (torchvision.transforms.Compose): 前処理関数
            location (str | os.PathLike, optional): データセットのルートディレクトリ. \
                Defaults to os.path.expanduser("dataset").
            batch_size (int, optional): バッチサイズ. Defaults to 32.
            num_workers (int, optional): データローダーの並列数. Defaults to 4.
        """
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), "train")
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {"shuffle": True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), "val_in_folder")
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), "val")
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return "imagenet"


class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass


class ImageNetK(ImageNet):

    def get_train_sampler(self):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:self.k()] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = ImageNet(preprocess, location=root)
    for i, batch in enumerate(d.train_loader):
        print(batch["images"].shape, batch["labels"], batch["image_paths"])
        break
