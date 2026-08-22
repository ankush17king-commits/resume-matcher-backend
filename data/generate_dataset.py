"""
generate_dataset.py (v2)

v1 generated short "Skills: X, Y, Z." snippets. That trained a model
that looked fine on paper (66% accuracy) but broke on real resumes/JDs,
because real text is long, written in prose, full of section headers
("Responsibilities", "Requirements"), boilerplate phrases, and words
that never appeared in the narrow training vocabulary. The model was
extrapolating wildly out-of-distribution on real input.

Fix: generate resumes/JDs that actually LOOK like the real thing --
multi-sentence, section-structured, with skills embedded naturally in
sentences alongside generic English (not just a comma-separated list).
This makes the TF-IDF vocabulary and length distributions match what
the model will actually see at inference time.
"""

import os
import random
import pandas as pd

random.seed(42)

# Always save next to this script (data/synthetic_dataset.csv), regardless
# of which directory this script is run FROM -- otherwise running it as
# `python3 data/generate_dataset.py` from the project root silently writes
# the CSV into the root folder instead of data/, and train_classifier.py
# ends up training on a stale/old dataset without any error or warning.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_PATH = os.path.join(SCRIPT_DIR, "synthetic_dataset.csv")

DOMAIN_SKILLS = {
    "web_dev": [
        "React", "JavaScript", "Node.js", "Express.js", "REST APIs",
        "MongoDB", "HTML5", "CSS3", "JWT authentication", "Redux",
        "TypeScript", "Vite", "responsive design", "Git"
    ],
    "data_science": [
        "Python", "pandas", "NumPy", "scikit-learn", "TensorFlow",
        "data visualization", "SQL", "machine learning", "statistics",
        "Jupyter", "feature engineering", "regression models", "NLP"
    ],
    "android": [
        "Kotlin", "Jetpack Compose", "Android Studio", "Room DB",
        "MVVM architecture", "Coroutines", "Dagger Hilt", "Firebase",
        "Material Design", "Java", "Android SDK"
    ],
    "devops": [
        "Docker", "Kubernetes", "CI/CD", "AWS", "Jenkins", "Terraform",
        "Linux administration", "Ansible", "monitoring", "Git",
        "shell scripting", "Azure"
    ],
    "design": [
        "Figma", "UI/UX design", "wireframing", "prototyping",
        "user research", "design systems", "Adobe XD", "Photoshop",
        "accessibility", "typography"
    ],
}

ROLE_TITLES = {
    "web_dev": ["Full Stack Developer", "Frontend Engineer", "Web Developer", "Software Engineer"],
    "data_science": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "ML Engineer"],
    "android": ["Android Developer", "Mobile Engineer", "Android Software Engineer"],
    "devops": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Infrastructure Engineer"],
    "design": ["Product Designer", "UI/UX Designer", "Visual Designer"],
}

RESUME_INTROS = [
    "Software engineering graduate with hands-on project experience and a strong foundation in computer science fundamentals.",
    "Motivated developer with practical experience building and deploying applications end to end.",
    "Engineering student passionate about building scalable, real-world software solutions.",
    "Detail-oriented developer with a track record of shipping projects independently and in teams.",
]

RESUME_EXPERIENCE_TEMPLATES = [
    "Built and maintained applications using {skills}, collaborating with a small team to ship features on schedule.",
    "Worked on a project involving {skills}, focusing on clean architecture and maintainable code.",
    "Developed features end to end using {skills}, including testing and code review.",
    "Gained hands-on experience with {skills} through internship and personal projects.",
]

RESUME_CLOSERS = [
    "Comfortable working independently and under tight deadlines.",
    "Strong communicator with experience collaborating in agile, cross-functional teams.",
    "Quick learner, eager to take ownership of new problems.",
    "Enjoys mentoring peers and contributing to open-source projects in spare time.",
]

JD_INTROS = [
    "We are looking for a {role} to join our growing engineering team.",
    "Our company is hiring a {role} to help build and scale our core product.",
    "We're seeking an experienced {role} to work closely with our product and engineering teams.",
    "Join us as a {role} and help shape the next generation of our platform.",
]

JD_RESPONSIBILITY_TEMPLATES = [
    "Design, build, and maintain systems using {skills}.",
    "Collaborate with cross-functional teams to deliver features involving {skills}.",
    "Write clean, well-tested, maintainable code using {skills}.",
    "Participate in code reviews, sprint planning, and technical design discussions.",
    "Own projects end to end, from design through deployment and monitoring.",
]

JD_REQUIREMENT_TEMPLATES = [
    "Strong proficiency in {skills}.",
    "Hands-on experience with {skills}.",
    "Familiarity with {skills} and modern development practices.",
    "Bachelor's degree in Computer Science or equivalent practical experience.",
    "Excellent communication skills and ability to work in a collaborative team environment.",
]

JD_NICE_TO_HAVE = [
    "Experience with {skills} is a plus.",
    "Exposure to {skills} is beneficial but not required.",
]


def skills_phrase(skills):
    if len(skills) == 1:
        return skills[0]
    return ", ".join(skills[:-1]) + f", and {skills[-1]}"


def sample_skills(domain, n_primary=5, n_noise=1):
    primary = random.sample(DOMAIN_SKILLS[domain], k=min(n_primary, len(DOMAIN_SKILLS[domain])))
    other_domains = [d for d in DOMAIN_SKILLS if d != domain]
    noise_domain = random.choice(other_domains)
    noise = random.sample(DOMAIN_SKILLS[noise_domain], k=min(n_noise, len(DOMAIN_SKILLS[noise_domain])))
    return primary, noise


def build_resume(domain):
    primary, noise = sample_skills(domain, n_primary=6, n_noise=1)
    all_skills_pool = primary + noise
    random.shuffle(all_skills_pool)

    intro = random.choice(RESUME_INTROS)
    n_exp_sentences = random.randint(2, 3)
    exp_sentences = []
    for _ in range(n_exp_sentences):
        template = random.choice(RESUME_EXPERIENCE_TEMPLATES)
        chunk = random.sample(all_skills_pool, k=min(random.randint(2, 4), len(all_skills_pool)))
        exp_sentences.append(template.format(skills=skills_phrase(chunk)))
    closer = random.choice(RESUME_CLOSERS)

    skills_line = "Technical Skills: " + ", ".join(all_skills_pool) + "."

    text = " ".join([intro] + exp_sentences + [closer, skills_line])
    return text, set(primary)


def build_jd(domain):
    primary, noise = sample_skills(domain, n_primary=5, n_noise=1)
    all_skills_pool = primary + noise
    random.shuffle(all_skills_pool)

    role = random.choice(ROLE_TITLES[domain])
    intro = random.choice(JD_INTROS).format(role=role)

    n_resp = random.randint(3, 4)
    resp_sentences = []
    for _ in range(n_resp):
        template = random.choice(JD_RESPONSIBILITY_TEMPLATES)
        if "{skills}" in template:
            chunk = random.sample(all_skills_pool, k=min(random.randint(1, 3), len(all_skills_pool)))
            resp_sentences.append(template.format(skills=skills_phrase(chunk)))
        else:
            resp_sentences.append(template)

    n_req = random.randint(3, 4)
    req_sentences = []
    for _ in range(n_req):
        template = random.choice(JD_REQUIREMENT_TEMPLATES)
        if "{skills}" in template:
            chunk = random.sample(all_skills_pool, k=min(random.randint(1, 3), len(all_skills_pool)))
            req_sentences.append(template.format(skills=skills_phrase(chunk)))
        else:
            req_sentences.append(template)

    nice_to_have = ""
    if random.random() < 0.6:
        other_domain = random.choice([d for d in DOMAIN_SKILLS if d != domain])
        extra_skill = random.choice(DOMAIN_SKILLS[other_domain])
        nice_to_have = " " + random.choice(JD_NICE_TO_HAVE).format(skills=extra_skill)

    text = (
        f"{intro} Responsibilities: " + " ".join(resp_sentences) +
        " Requirements: " + " ".join(req_sentences) + nice_to_have
    )
    return text, set(primary)


def generate_pair():
    resume_domain = random.choice(list(DOMAIN_SKILLS.keys()))
    resume_text, resume_primary = build_resume(resume_domain)

    same_domain = random.random() < 0.5
    jd_domain = resume_domain if same_domain else random.choice(
        [d for d in DOMAIN_SKILLS if d != resume_domain]
    )
    jd_text, jd_primary = build_jd(jd_domain)

    overlap = len(resume_primary & jd_primary)
    union = len(resume_primary | jd_primary)
    jaccard = overlap / union if union else 0

    base_prob = 0.15 + 0.75 * jaccard
    if same_domain:
        base_prob += 0.1
    base_prob = min(base_prob, 0.97)
    label = 1 if random.random() < base_prob else 0

    return resume_text, jd_text, label


def main(n_samples=1500, out_path=None):
    if out_path is None:
        out_path = DEFAULT_OUT_PATH
    rows = [generate_pair() for _ in range(n_samples)]
    df = pd.DataFrame(rows, columns=["resume_text", "jd_text", "label"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["label"].value_counts(normalize=True))
    print("\nSample resume:\n", df.iloc[0]["resume_text"])
    print("\nSample JD:\n", df.iloc[0]["jd_text"])


if __name__ == "__main__":
    main()