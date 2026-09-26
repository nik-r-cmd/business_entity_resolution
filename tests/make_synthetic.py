"""Generate a SMALL synthetic dataset in the challenge format, only to smoke-test the pipeline end to end.
    python tests/make_synthetic.py --out synthetic/dataset
Not the real data and not meant for tuning - real noise is different."""
import argparse
import os
import random

W = {
    "US": (["Acme", "Summit", "Pioneer", "Liberty", "Eagle", "Golden", "Metro", "Prime", "Apex", "Harbor", "Cedar", "Maple", "Iron", "Blue", "Red"],
           ["Plumbing", "Foods", "Motors", "Consulting", "Logistics", "Dental", "Bakery", "Systems", "Realty", "Electric"],
           ["Inc", "LLC", "Corp", "Ltd"]),
    "India": (["Sri", "Shree", "Lakshmi", "Balaji", "Ganesh", "Venkateswara", "Bharat", "Hindustan", "Reddy", "Kumar", "Sai", "Annapurna"],
              ["Traders", "Enterprises", "Textiles", "Foods", "Constructions", "Pharma", "Electronics", "Agencies"],
              ["Pvt Ltd", "Private Limited", "& Sons", "Co"]),
    "France": (["Boulangerie", "Maison", "Atelier", "Garage", "Cafe", "Pharmacie", "Chez", "Domaine"],
               ["Dupont", "Martin", "Bernard", "Moreau", "Lefevre", "Girard", "Rousseau", "Fournier"],
               ["SARL", "SAS", "EURL"]),
}
STREETS = ["Main", "Oak", "Park", "Lake", "Hill", "Station", "Market", "Temple", "Church", "Garden", "River"]
CITIES = {"US": ["Austin", "Denver", "Boston", "Seattle"], "India": ["Hyderabad", "Pune", "Chennai", "Jaipur"], "France": ["Lyon", "Nantes", "Lille", "Rennes"]}
ABBR = [("Corporation", "Corp"), ("Limited", "Ltd"), ("Private", "Pvt"), ("Incorporated", "Inc"), ("Company", "Co"), ("&", "and")]
AABBR = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("rue", "r.")]


def typo(s, rng):
    if len(s) > 4 and rng.random() < 0.5:
        i = rng.randrange(1, len(s) - 2)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


def gen_entity(c, rng):
    a, b, suf = W[c]
    name = f"{rng.choice(a)} {rng.choice(b)} {rng.choice(suf)}" if c != "France" else f"{rng.choice(a)} {rng.choice(b)} {rng.choice(suf)}"
    city = rng.choice(CITIES[c])
    if c == "US":
        addr = f"{rng.randint(1, 999)} {rng.choice(STREETS)} {rng.choice(['Road', 'Street', 'Avenue'])}, {city}, {rng.randint(10000, 99999)}"
    elif c == "India":
        addr = f"{rng.randint(1, 99)}-{rng.randint(1, 99)}, {rng.choice(STREETS)} Road, {city}, {rng.randint(500000, 599999)}"
    else:
        addr = f"{rng.randint(1, 99)} rue {rng.choice(STREETS)}, {rng.randint(10000, 99999)} {city}"
    return name, addr


def noisy(name, addr, c, rng, level):
    for long, short in ABBR:
        if rng.random() < 0.5 * level:
            name = name.replace(long, short).replace(short, short)
    if rng.random() < 0.3 * level:
        name = typo(name, rng)
    if rng.random() < 0.2 * level:
        w = name.split()
        rng.shuffle(w)
        name = " ".join(w)
    if rng.random() < 0.3 * level:
        name = name.replace(" Inc", "").replace(" LLC", "").replace(" SARL", "")
    for long, short in AABBR:
        if rng.random() < 0.6:
            addr = addr.replace(long, short)
    if rng.random() < 0.3 * level:
        addr = ", ".join(addr.split(", ")[:-1]) or addr        # drop last component (pin/zip/city)
    if rng.random() < 0.25 * level and c == "India":
        addr += ", Near SBI ATM"
    if rng.random() < 0.05 * level:
        addr = ""
    return (name.upper() if rng.random() < 0.2 else name), addr


def make_split(name, countries, n_per_country, rng, out):
    s1, s2, s3, gt = [], [], [], []
    n1 = n2 = n3 = 0
    for c in countries:
        for _ in range(n_per_country):
            nm, ad = gen_entity(c, rng)
            n1 += 1
            sid = f"S1-{n1:05d}"
            s1.append((sid, nm, ad, c))
            m = []
            if rng.random() > 0.15:                      # 15% singletons
                for _ in range(rng.choice([0, 1, 1, 2])):
                    n2 += 1
                    t = f"S2-{n2:05d}"
                    x, y = noisy(nm, ad, c, rng, 1.0)
                    s2.append((t, x, y, c))
                    m.append(t)
                for _ in range(rng.choice([0, 1, 1, 2])):
                    n3 += 1
                    t = f"S3-{n3:05d}"
                    x, y = noisy(nm, ad, c, rng, 1.5)
                    s3.append((t, x, y, c))
                    m.append(t)
            gt.append((sid, ",".join(m)))
        for _ in range(int(n_per_country * 0.25)):       # orphan distractors
            nm, ad = gen_entity(c, rng)
            n2 += 1
            s2.append((f"S2-{n2:05d}", *noisy(nm, ad, c, rng, 1.0), c))
            nm, ad = gen_entity(c, rng)
            n3 += 1
            s3.append((f"S3-{n3:05d}", *noisy(nm, ad, c, rng, 1.0), c))
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for i, rows in enumerate([s1, s2, s3], 1):
        rng.shuffle(rows) if i > 1 else None
        with open(f"{d}/{name}_source{i}.tsv", "w") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
    if name == "train":
        with open(f"{d}/train_ground_truth.tsv", "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for r in gt:
                f.write(f"{r[0]}\t{r[1]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic/dataset")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    rng = random.Random(0)
    make_split("train", ["US", "India"], a.n, rng, a.out)
    make_split("test", ["US", "India", "France"], a.n // 2, rng, a.out)
    print("synthetic data written to", a.out)
