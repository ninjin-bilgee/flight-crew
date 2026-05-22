from preprocessing import filter_jd_sections

fake_jd = """
Required
Communication skills
Team collaboration

Skills to Gain
Python
SQL
Machine Learning

Responsibilities
Work with internal teams
"""

filtered = filter_jd_sections(fake_jd)

print(filtered)