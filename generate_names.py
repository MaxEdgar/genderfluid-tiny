#!/usr/bin/env python3
"""
Generate a 1000-name synthetic dataset.

SYNTHETIC DATA - NOT real-world evidence.
Created for pipeline testing and demonstration purposes only.
Names are based on common naming patterns but are NOT sourced from real census data.
"""

import json
import os
import random

GIRL_NAMES = [
    # Classic English
    "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte",
    "Amelia", "Harper", "Evelyn", "Abigail", "Emily", "Ella", "Elizabeth",
    "Camila", "Luna", "Sofia", "Aria", "Scarlett", "Penelope", "Layla",
    "Chloe", "Victoria", "Madison", "Eleanor", "Grace", "Nora", "Riley",
    "Zoey", "Hannah", "Hazel", "Lily", "Ellie", "Violet", "Aurora",
    "Savannah", "Audrey", "Brooklyn", "Bella", "Claire", "Skylar", "Lucy",
    "Paisley", "Anna", "Caroline", "Nova", "Genesis", "Emilia", "Kennedy",
    "Samantha", "Maya", "Willow", "Kinsley", "Naomi", "Aaliyah", "Elena",
    "Sarah", "Ariana", "Allison", "Gabriella", "Alice", "Madelyn", "Cora",
    "Ruby", "Eva", "Serenity", "Autumn", "Adeline", "Hailey", "Gianna",
    "Quinn", "Natalie", "Aubrey", "Josephine", "Rylee", "Arianna", "Finley",
    "Lillian", "Melanie", "Daniella", "Lydia", "Vivian", "Lauren", "Maria",
    "Jasmine", "Mary", "Iris", "Ivy", "Jade", "Elsie", "Melody",
    "Leah", "Piper", "Rosalie", "Marie", "Willa", "Margaret", "Danielle",
    "Elva", "Retta", "Michelle", "Renatta", "Priya", "Yuki", "Mei",
    "Fatima", "Amara", "Zara", "Clara", "Stella", "Natalia", "Brianna",
    "Ashley", "Katherine", "Alexis", "Lillian", "Hannah", "Makayla",
    # More feminine names
    "Bianca", "Carmen", "Celeste", "Celia", "Celine", "Charlotte", "Cherry",
    "Christina", "Cindy", "Clementine", "Colette", "Columbia", "Connie",
    "Constanza", "Cora", "Cynthia", "Daisy", "Daphne", "Dawn", "Deborah",
    "Debra", "Denise", "Diana", "Diane", "Dolores", "Donna", "Doris",
    "Dorothy", "Edith", "Eileen", "Elaine", "Elena", "Ellen", "Eloise",
    "Emily", "Emma", "Erica", "Erin", "Esther", "Ethel", "Eugenie",
    "Eunice", "Valentina", "Valeria", "Vanessa", "Vera", "Veronica",
    "Victoria", "Violet", "Virginia", "Vivian", "Wanda", "Wendy",
    "Whitney", "Wilhelmina", "Winifred", "Yolanda", "Yvette", "Yvonne",
    "Zelda", "Zenobia", "Zephyrine", "Zoe", "Zola", "Zora",
    # International feminine
    "Akiko", "Aiko", "Sakura", "Haruka", "Mizuki", "Rin", "Hana",
    "Soo", "Minji", "Jisoo", "Nari", "Eunji", "Dayoung",
    "Ananya", "Deepa", "Kavya", "Lakshmi", "Meera", "Nisha", "Pooja",
    "Priyanka", "Radha", "Rani", "Sita", "Sunita", "Trisha", "Vidya",
    "Amira", "Basmala", "Farah", "Hana", "Layla", "Malika", "Nadia",
    "Nadia", "Salma", "Yasmin", "Zahra",
    "Ingrid", "Astrid", "Freya", "Sigrid", "Elsie", "Thora",
    "Anouk", "Fleur", "Lotte", "Saskia", "Wren",
    "Adriana", "Alejandra", "Beatriz", "Camila", "Carmen", "Clara",
    "Constanza", "Daniela", "Elena", "Fernanda", "Gabriela", "Isabela",
    "Julieta", "Lucia", "Mariana", "Natalia", "Paula", "Renata",
    "Sofia", "Valentina", "Ximena",
    "Aaliyah", "Amira", "Anaya", "Azaria", "Brielle", "Cadence",
    "Callie", "Catalina", "Charli", "Dahlia", "Davina", "Elise",
    "Eliza", "Ellianna", "Ember", "Emersyn", "Estrella", "Everly",
    "Fatima", "Gia", "Giovanna", "Gracie", "Harmony", "Harlow",
    "Hayden", "Hope", "Isla", "Ivory", "Juliette", "Kailani",
    "Kaia", "Kamila", "Kara", "Kehlani", "Leila", "Lena",
    "Lia", "Liliana", "Lois", "London", "Lylah", "Mabel",
    "Maci", "Mackenzie", "Magnolia", "Maisie", "Maren", "Marley",
    "Martha", "Mila", "Millie", "Molly", "Morgan", "Nadia",
    "Nathalie", "Navy", "Nia", "Nyla", "Olive", "Paisley",
    "Pearl", "Peyton", "Phoenix", "Piper", "Posie", "Rae",
    "Reese", "River", "Robin", "Rory", "Rosie", "Rowan",
    "Royal", "Sage", "Selena", "Serena", "Skye", "Sloane",
    "Sophie", "Stevie", "Summer", "Sunny", "Sydney", "Tessa",
    "Tiana", "Tiffany", "Valentina", "Valerie", "Vera", "Vivienne",
    "Winter", "Xiomara", "Zaria", "Zuri",
    # More from various cultures
    "Adela", "Adelina", "Adrienne", "Agatha", "Agnes", "Aimee",
    "Alana", "Alberta", "Alexa", "Alexandra", "Alina", "Aline",
    "Allegra", "Alma", "Amara", "Amber", "Amelia", "Amy",
    "Anastasia", "Andrea", "Angelina", "Annabelle", "Annika", "Antoinette",
    "Arabella", "Ariel", "Ayla", "Barbara", "Beatrice", "Belinda",
    "Bernadette", "Bethany", "Beverly", "Bridget", "Brooke", "Calista",
    "Cambria", "Camille", "Candace", "Carla", "Carla", "Catalina",
    "Cecilia", "Celeste", "Chanel", "Chantal", "Cher", "Cheryl",
    "Chloe", "Christine", "Cindy", "Claudia", "Clementine", "Colleen",
    "Constance", "Cordelia", "Crystal", "Cybil", "Cynthia", "Dahlia",
    "Daisy", "Daphne", "Darla", "Dawn", "Deanna", "Debbie",
    "Delia", "Delilah", "Demi", "Denise", "Destiny", "Diana",
    "Diane", "Dina", "Dolores", "Dominique", "Donna", "Doris",
    "Dorothy", "Dulce", "Easter", "Ebony", "Eden", "Edith",
    "Eileen", "Elaine", "Elena", "Eliana", "Elisa", "Elizabeth",
    "Ella", "Ellie", "Emerson", "Emery", "Emily", "Emma",
    "Erica", "Erika", "Erin", "Erma", "Esmeralda", "Estelle",
    "Esther", "Etsy", "Ethel", "Eugenia", "Eunice", "Eva",
    "Evangeline", "Eve", "Evelyn", "Everleigh", "Evie", "Faith",
    "Fanny", "Fatima", "Felicia", "Felicity", "Fern", "Fiona",
    "Flora", "Florence", "Frances", "Francesca", "Francine", "Freya",
    "Gabrielle", "Gail", "Gemma", "Genevieve", "Gentle", "Georgia",
    "Geraldine", "Gertrude", "Giana", "Gianna", "Ginger", "Giselle",
    "Gloria", "Goldie", "Grace", "Gracelyn", "Gracie", "Greta",
    "Gwen", "Gwendolyn", "Haley", "Hallie", "Hana", "Hannah",
    "Harlee", "Harley", "Harmony", "Harper", "Harriet", "Haven",
    "Hayley", "Hazel", "Heidi", "Helen", "Helena", "Helene",
    "Hera", "Hope", "Hugh", "Ianthe", "Ila", "Ilana",
    "Ileana", "Ilene", "Ilona", "Imani", "Imogen", "Ina",
    "India", "Indie", "Ines", "Ingrid", "Irene", "Iris",
    "Irma", "Isabel", "Isabella", "Isabelle", "Isadora", "Isolde",
    "Itzel", "Ivana", "Ivy", "Ivory", "Ivy", "Jacqueline",
    "Jade", "Jade", "Jane", "Janessa", "Janet", "Janice",
    "Jean", "Jeanette", "Jemma", "Jenna", "Jennifer", "Jenny",
    "Jerica", "Jessie", "Jill", "Jo", "Joan", "Joanna",
    "Jocelyn", "Joelle", "Johanna", "Jolene", "Jolie", "Journey",
    "Joy", "Joyce", "Judith", "Julia", "Juliana", "Julianne",
    "Julie", "Juliet", "Juliette", "June", "Juniper", "Justine",
]

BOY_NAMES = [
    # Classic English
    "James", "Robert", "John", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric",
    "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Benjamin",
    "Samuel", "Raymond", "Gregory", "Frank", "Alexander", "Patrick", "Jack",
    "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Adam", "Nathan",
    "Henry", "Zachary", "Douglas", "Peter", "Noah", "Ethan", "Liam",
    "Mason", "Logan", "Lucas", "Oliver", "Aiden", "Max", "Leo",
    "Jackson", "Sebastian", "Mateo", "Owen", "Elijah", "Grayson",
    "Marcus", "Terrence", "Kwame", "Dmitri", "Hiroshi", "Javier",
    "Ahmed", "Raj", "Omar", "Ali", "Chen", "Hans", "Pierre",
    # More masculine names
    "Adrian", "Albert", "Alfred", "Allan", "Andre", "Antonio",
    "Arnold", "Arthur", "Austin", "Barry", "Barton", "Basil",
    "Ben", "Bernard", "Bert", "Billy", "Blake", "Bobby",
    "Boris", "Brad", "Bradley", "Brent", "Brett", "Brian",
    "Bruce", "Bryan", "Caleb", "Carl", "Carlos", "Carter",
    "Cedric", "Chad", "Charles", "Chester", "Chris", "Clarence",
    "Claude", "Clayton", "Clyde", "Colin", "Conrad", "Corey",
    "Cornelius", "Craig", "Curt", "Curtis", "Cyrus", "Dale",
    "Damian", "Damon", "Dan", "Dane", "Darrell", "Darren",
    "Dave", "David", "Dean", "Delbert", "Dennis", "Derek",
    "Derrick", "Desmond", "Dexter", "Dick", "Diego", "Dillon",
    "Dominic", "Don", "Donald", "Donnie", "Dorian", "Doug",
    "Douglas", "Drew", "Dustin", "Dwayne", "Dylan", "Earl",
    "Ed", "Eddie", "Edgar", "Edmund", "Eduard", "Edward",
    "Edwin", "Eli", "Elias", "Elijah", "Elliot", "Elliott",
    "Elmer", "Emerson", "Emmett", "Enrique", "Erik", "Ernest",
    "Eugene", "Evan", "Felix", "Ferdinand", "Fernando", "Finn",
    "Floyd", "Ford", "Forrest", "Francis", "Francisco", "Frank",
    "Franklin", "Frederick", "Fritz", "Gabriel", "Gail", "Gavin",
    "Gene", "Geoffrey", "George", "Gerald", "Gerard", "Gilbert",
    "Glen", "Glenn", "Gordon", "Graham", "Grant", "Grayson",
    "Greg", "Gregg", "Gregory", "Gus", "Guy", "Harold",
    "Harry", "Harvey", "Heath", "Hector", "Henry", "Herbert",
    "Herman", "Homer", "Howard", "Hugh", "Hugo", "Humberto",
    "Ian", "Ira", "Isaiah", "Ivan", "Ivor", "Jack",
    "Jackson", "Jacob", "Jake", "James", "Jamie", "Jared",
    "Jason", "Javier", "Jay", "Jean", "Jeff", "Jeffery",
    "Jeffrey", "Jerald", "Jeremiah", "Jeremy", "Jerome", "Jesse",
    "Jimmy", "Joaquin", "Joe", "Joel", "John", "Johnny",
    "Jon", "Jonah", "Jonathan", "Jordan", "Jorge", "Jose",
    "Joseph", "Josh", "Joshua", "Juan", "Julian", "Julius",
    "June", "Junior", "Justin", "Kaleb", "Karl", "Keith",
    "Kelly", "Kenneth", "Kent", "Kerry", "Kevin", "Kirk",
    "Kyle", "Lance", "Larry", "Laurence", "Lawrence", "Lee",
    "Leon", "Leonard", "Leroy", "Leslie", "Lester", "Liam",
    "Lindsey", "Lionel", "Lloyd", "Logan", "Lonnie", "Louis",
    "Louie", "Lucas", "Luis", "Luke", "Luther", "Lyle",
    "Mack", "Malcolm", "Manuel", "Marc", "Marcel", "Marcus",
    "Mario", "Marion", "Mark", "Marshall", "Martin", "Mason",
    "Mateo", "Mathew", "Matt", "Matthew", "Maurice", "Max",
    "Melvin", "Merle", "Micah", "Michael", "Micheal", "Miguel",
    "Mike", "Miles", "Mitchell", "Morris", "Moses", "Nathan",
    "Nathaniel", "Neil", "Nelson", "Nicholas", "Nick", "Noah",
    "Nolan", "Norman", "Oliver", "Omar", "Orlando", "Oscar",
    "Otis", "Owen", "Pablo", "Parker", "Pat", "Patrick",
    "Paul", "Pedro", "Perry", "Pete", "Peter", "Phil",
    "Philip", "Phillip", "Quentin", "Rafael", "Ralph", "Ramiro",
    "Randall", "Randy", "Raphael", "Raul", "Ray", "Raymond",
    "Reggie", "Reginald", "Rex", "Reynold", "Richard", "Rick",
    "Rickey", "Ricky", "Riley", "Rob", "Robert", "Robin",
    "Rod", "Rodney", "Rodrigo", "Roger", "Roland", "Roman",
    "Ron", "Ronald", "Ronnie", "Roosevelt", "Ross", "Roy",
    "Ruben", "Rudolf", "Rudy", "Russ", "Russell", "Rusty",
    "Ryan", "Salvador", "Sam", "Samuel", "Santiago", "Santos",
    "Saul", "Scott", "Sean", "Sergio", "Seth", "Shane",
    "Shawn", "Sheldon", "Sidney", "Silas", "Spencer", "Stan",
    "Stanley", "Stefan", "Stephen", "Steve", "Steven", "Stewart",
    "Stuart", "Sylvester", "Ted", "Terence", "Terrence", "Terry",
    "Theodore", "Thomas", "Tim", "Timothy", "Todd", "Tom",
    "Tommy", "Tony", "Tracey", "Travis", "Trent", "Trevor",
    "Tristan", "Troy", "Truman", "Turner", "Tyler", "Ulrich",
    "Val", "Vernon", "Victor", "Vincent", "Vinson", "Virgil",
    "Vito", "Wade", "Walker", "Wallace", "Walter", "Ward",
    "Warren", "Wayne", "Wesley", "Wilbur", "Willard", "William",
    "Willie", "Willy", "Wyatt",
    # International masculine
    "Akira", "Kenji", "Takeshi", "Yuto", "Haruto", "Sota",
    "Minho", "Jaehyun", "Taeyong", "Junseo", "Dohyun",
    "Arjun", "Rohan", "Vikram", "Sanjay", "Rajesh", "Amit",
    "Hassan", "Karim", "Youssef", "Tariq", "Rashid", "Ibrahim",
    "Sven", "Erik", "Anders", "Bjorn", "Lars", "Gunnar",
    "Pierre", "Jean", "Andre", "Jacques", "Marcel", "Rene",
    "Carlos", "Miguel", "Pedro", "Diego", "Fernando", "Rafael",
    "Lars", "Hans", "Fritz", "Otto", "Klaus", "Werner",
    "Dmitri", "Alexei", "Boris", "Sergei", "Nikolai", "Vladimir",
    "Omar", "Ali", "Hassan", "Khalil", "Tariq", "Jamal",
    "Kofi", "Kwame", "Emeka", "Chidi", "Tendai", "Bongani",
    "Bruno", "Enzo", "Marco", "Leonardo", "Matteo", "Raffaele",
    "Inigo", "Javier", "Alonso", "Sergio", "Pablo", "Arturo",
]

UNCERTAIN_NAMES = [
    "Alex", "Sam", "Taylor", "Jordan", "Chris", "Pat", "Jamie",
    "Casey", "Morgan", "Riley", "Quinn", "Dakota", "Reese", "Skyler",
    "Peyton", "Finley", "Hayden", "Emerson", "Rowan", "Sage", "River",
    "Phoenix", "Robin", "Kai", "Nico", "Avery", "Cameron", "Drew",
    "Jessie", "Leslie", "Marion", "Ren", "Sora", "Blake", "Devin",
    "Dylan", "Ellis", "Harley", "Harper", "Kendall", "Lane",
    "Lee", "London", "Lyric", "Mackenzie", "Madison", "Micah",
    "Oakley", "Parker", "Raven", "Reagan", "Shannon", "Shawn",
    "Sky", "Spencer", "Stevie", "Tatum", "Terry", "Tony",
    "Tracy", "Wren", "Zion", "Adrian", "Arden", "Ashley",
    "Aubrey", "Bay", "Becky", "Bernie", "Billy", "Bobby",
    "Carmen", "Charlie", "Conner", "Courtney", "Danny", "Devon",
    "Eden", "Elliot", "Erin", "Florence", "Frances", "Glenn",
    "Hollis", "Hurley", "Jody", "Justice", "Keegan",
    "Kelly", "Kim", "Kris", "Lane", "Lindsey",
    "Max", "Nicole", "Noel", "Peyton", "Reagan",
    "Rene", "Robin", "Rowan", "Sage", "Shannon",
    "Shawn", "Sky", "Stevie", "Tatum", "Tracy",
    "Wren", "Zion", "Carey", "Dana", "Dee",
    "Dell", "Erin", "Gene", "Glen", "Hyde",
    "Jean", "Jules", "Kerry", "Kim", "Lane",
    "Leslie", "Lindsey", "Lou", "Max", "Micah",
    "Noel", "Peyton", "Reagan", "Rene", "Robin",
    "Rowan", "Sage", "Shannon", "Shawn", "Sky",
    "Spencer", "Stevie", "Tatum", "Tracy", "Val",
    "Wren", "Zion", "Dakota", "Eden", "Emery",
    "Harper", "Jesse", "Jordan", "Kendall", "Lane",
    "Lee", "Leslie", "Lindsey", "London", "Lou",
    "Lyric", "Mackenzie", "Madison", "Marion", "Max",
    "Micah", "Noel", "Oakley", "Parker", "Peyton",
    "Reagan", "Rene", "Robin", "Rowan", "Sage",
    "Shannon", "Shawn", "Sky", "Spencer", "Stevie",
    "Tatum", "Terry", "Tracy", "Val", "Wren",
    "Zion", "Carey", "Dana", "Dee", "Dell",
    "Glen", "Jules", "Kerry",
]


def main():
    random.seed(42)

    entries = []

    # Girl names
    for name in GIRL_NAMES:
        entries.append({"name": name, "label": "girl-associated", "weight": 1.0})

    # Boy names
    for name in BOY_NAMES:
        entries.append({"name": name, "label": "boy-associated", "weight": 1.0})

    # Uncertain names
    for name in UNCERTAIN_NAMES:
        entries.append({"name": name, "label": "uncertain", "weight": 1.0})

    # Deduplicate (some names may appear in multiple lists)
    seen = set()
    unique_entries = []
    for entry in entries:
        key = entry["name"].lower()
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)

    # If we have fewer than 1000, generate variations
    if len(unique_entries) < 1000:
        # Generate compound names
        girl_first = ["Emma", "Olivia", "Ava", "Sophia", "Mia", "Isabella", "Charlotte",
                      "Amelia", "Harper", "Evelyn", "Luna", "Sofia", "Aria", "Chloe",
                      "Victoria", "Grace", "Nora", "Lily", "Hannah", "Zoey"]
        girl_last = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                     "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor",
                     "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
                     "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis",
                     "Robinson", "Walker", "Young", "Allen", "King", "Wright",
                     "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
                     "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
                     "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans",
                     "Turner", "Diaz", "Parker", "Cruz", "Edwards", "Collins",
                     "Reyes", "Stewart", "Morris", "Morales", "Murphy", "Cook",
                     "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson",
                     "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim",
                     "Cox", "Ward", "Richardson", "Watson", "Brooks", "Chavez",
                     "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz",
                     "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel",
                     "Myers", "Long", "Ross", "Foster", "Jimenez", "Powell",
                     "Jenkins", "Perry", "Russell", "Sullivan", "Bell", "Coleman",
                     "Butler", "Henderson", "Barnes", "Gonzales", "Fisher", "Vasquez"]

        boy_first = ["James", "Michael", "William", "David", "Joseph", "Thomas",
                     "Christopher", "Daniel", "Matthew", "Anthony", "Andrew", "Joshua",
                     "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward",
                     "Jason", "Jeffrey", "Ryan", "Jacob", "Nicholas", "Eric",
                     "Jonathan", "Stephen", "Justin", "Scott", "Brandon", "Benjamin",
                     "Samuel", "Raymond", "Gregory", "Alexander", "Patrick", "Jack",
                     "Dennis", "Jerry", "Tyler", "Aaron", "Adam", "Nathan",
                     "Henry", "Zachary", "Douglas", "Peter", "Noah", "Ethan",
                     "Liam", "Mason", "Logan", "Lucas", "Oliver", "Aiden"]
        boy_last = girl_last[:]  # Same surnames

        # Generate girl full names
        for i in range(200):
            first = random.choice(girl_first)
            last = random.choice(girl_last)
            full = f"{first} {last}"
            key = full.lower()
            if key not in seen:
                seen.add(key)
                unique_entries.append({"name": full, "label": "girl-associated", "weight": 1.0})

        # Generate boy full names
        for i in range(200):
            first = random.choice(boy_first)
            last = random.choice(boy_last)
            full = f"{first} {last}"
            key = full.lower()
            if key not in seen:
                seen.add(key)
                unique_entries.append({"name": full, "label": "boy-associated", "weight": 1.0})

        # Generate girl middle+last names
        middle_names = ["Marie", "Jane", "Anne", "Rose", "Lynn", "Grace", "Lee",
                        "Nicole", "Rae", "May", "Lou", "Mae", "Jo", "Fae"]
        for i in range(50):
            first = random.choice(girl_first)
            middle = random.choice(middle_names)
            last = random.choice(girl_last)
            full = f"{first} {middle} {last}"
            key = full.lower()
            if key not in seen:
                seen.add(key)
                unique_entries.append({"name": full, "label": "girl-associated", "weight": 1.0})

        # Generate boy middle+last names
        middle_m = ["James", "Michael", "Robert", "David", "William", "John",
                    "Thomas", "Edward", "Paul", "Mark", "Ray", "Lee", "Wayne"]
        for i in range(50):
            first = random.choice(boy_first)
            middle = random.choice(middle_m)
            last = random.choice(boy_last)
            full = f"{first} {middle} {last}"
            key = full.lower()
            if key not in seen:
                seen.add(key)
                unique_entries.append({"name": full, "label": "boy-associated", "weight": 1.0})

    # Shuffle
    random.shuffle(unique_entries)

    # Limit to 1000
    unique_entries = unique_entries[:1000]

    # Write to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "names.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in unique_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print statistics
    girl_count = sum(1 for e in unique_entries if e["label"] == "girl-associated")
    boy_count = sum(1 for e in unique_entries if e["label"] == "boy-associated")
    uncertain_count = sum(1 for e in unique_entries if e["label"] == "uncertain")

    print(f"Generated {len(unique_entries)} unique names")
    print(f"  Girl-associated: {girl_count}")
    print(f"  Boy-associated: {boy_count}")
    print(f"  Uncertain: {uncertain_count}")
    print(f"\nSaved to: {output_path}")
    print("\nSYNTHETIC DATA - NOT real-world evidence.")


if __name__ == "__main__":
    main()
