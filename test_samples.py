"""
SIMPLE TEST SCRIPT FOR LEGAL TEXT CLASSIFICATION
Run this after starting app.py to test the system
"""

import requests
import json

# Change this to your ngrok URL after running app.py
BASE_URL = "http://localhost:5000"  # Change to your ngrok URL if testing remotely

print("🧪 TESTING LEGAL TEXT CLASSIFICATION SYSTEM")
print("=" * 60)

# Test Sample 1: Citation
print("\n📝 TEST 1: Citation Case")
print("-" * 60)
text1 = "In the case of Smith v. Jones (2020), the court cited the precedent set in Brown v. Board of Education regarding constitutional rights and equal protection under the law."
print(f"Input: {text1[:100]}...")

response1 = requests.post(f"{BASE_URL}/predict", json={"text": text1})
result1 = response1.json()
print(f"\n✅ Result: {result1['label'].upper()}")
print(f"📊 Confidence: {result1['confidence']}")

# Test Sample 2: Application
print("\n\n📝 TEST 2: Application of Doctrine")
print("-" * 60)
text2 = "The court applied the doctrine of res judicata, determining that the matter had already been adjudicated in a prior proceeding and therefore could not be relitigated."
print(f"Input: {text2[:100]}...")

response2 = requests.post(f"{BASE_URL}/predict", json={"text": text2})
result2 = response2.json()
print(f"\n✅ Result: {result2['label'].upper()}")
print(f"📊 Confidence: {result2['confidence']}")

# Test Sample 3: Following Precedent
print("\n\n📝 TEST 3: Following Precedent")
print("-" * 60)
text3 = "Following the established precedent in Miranda v. Arizona, the court ruled that the defendant's rights were violated when statements were obtained without proper warnings."
print(f"Input: {text3[:100]}...")

response3 = requests.post(f"{BASE_URL}/predict", json={"text": text3})
result3 = response3.json()
print(f"\n✅ Result: {result3['label'].upper()}")
print(f"📊 Confidence: {result3['confidence']}")

# Test Sample 4: Reference
print("\n\n📝 TEST 4: Reference to Conventions")
print("-" * 60)
text4 = "The judgment referred to multiple international conventions and treaties, including the Geneva Convention and the Universal Declaration of Human Rights."
print(f"Input: {text4[:100]}...")

response4 = requests.post(f"{BASE_URL}/predict", json={"text": text4})
result4 = response4.json()
print(f"\n✅ Result: {result4['label'].upper()}")
print(f"📊 Confidence: {result4['confidence']}")

# Get available categories
print("\n\n📊 AVAILABLE CATEGORIES:")
print("-" * 60)
if 'all_categories' in result1:
    for i, cat in enumerate(result1['all_categories'], 1):
        print(f"{i}. {cat.upper()}")

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED!")
print("=" * 60)
