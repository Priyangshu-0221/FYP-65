
import json
import random

existing_file_path = r'c:\Users\abhij\Desktop\Project\FYP-65\backend\data\dummy_internship_recommendations.json'
with open(existing_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

last_id = data[-1]['id']
id_counter = int(last_id.replace('INT', '')) + 1

mobile_jobs = [
    {
        "title": "Android Developer Intern",
        "company": "AppVantage Labs",
        "domain": "Mobile Development",
        "skills": ["android", "kotlin", "java", "android studio", "firebase"],
        "desc": "Join our mobile team to build high-performance Android applications. Work with Kotlin and Jetpack Compose."
    },
    {
        "title": "iOS Developer Intern",
        "company": "AppleTree Systems",
        "domain": "Mobile Development",
        "skills": ["ios", "swift", "xcode", "uikit", "swiftui"],
        "desc": "Help develop cutting-edge iOS applications for our global clients using the latest Apple technologies."
    },
    {
        "title": "Flutter Intern",
        "company": "CrossPlatform Co",
        "domain": "Mobile Development",
        "skills": ["flutter", "dart", "firebase", "api integration", "mobile design"],
        "desc": "Build beautiful cross-platform apps for Android and iOS using Flutter."
    },
    {
        "title": "React Native Developer Intern",
        "company": "Hybrid Apps Inc",
        "domain": "Mobile Development",
        "skills": ["react native", "javascript", "typescript", "mobile development", "redux"],
        "desc": "Work on a high-traffic mobile app used by millions. Master React Native and bridge native modules."
    },
    {
        "title": "Mobile App UI/UX Intern",
        "company": "DesignFirst Mobile",
        "domain": "Design",
        "skills": ["figma", "mobile design", "prototyping", "user research", "android"],
        "desc": "Focus on the visual and interactive aspects of mobile applications across platforms."
    },
    {
        "title": "Junior Android Developer",
        "company": "DroidWorld",
        "domain": "Mobile Development",
        "skills": ["android", "java", "sql", "git", "mvvm"],
        "desc": "Support the maintenance and feature development of our core Android product."
    }
]

locations = ["Bangalore, India", "Mumbai, India", "Hyderabad, India", "Remote", "Pune, India"]
stipends = ["₹15,000/month", "₹20,000/month", "₹25,000/month"]

for job in mobile_jobs:
    # Add 2 variations of each
    for _ in range(2):
        loc = random.choice(locations)
        data.append({
            "id": f"INT{id_counter:03d}",
            "title": job["title"],
            "company": job["company"] + (" Group" if _ == 1 else " Inc"),
            "location": loc,
            "domain": job["domain"],
            "skills_required": job["skills"],
            "description": f"{job['desc']} Based in {loc}.",
            "apply_link": f"https://example.com/apply/int{id_counter:03d}",
            "duration": "6 months",
            "stipend": random.choice(stipends)
        })
        id_counter += 1

with open(existing_file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Added {len(mobile_jobs) * 2} mobile jobs. Total: {len(data)}")
