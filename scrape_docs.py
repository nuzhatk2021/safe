import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# All the sites we want to pull from
sources = [
    "https://store.samhsa.gov/sites/default/files/sma14-4816.pdf",
    "https://store.samhsa.gov/sites/default/files/pep20-01-01-001.pdf",
    "https://www.nimh.nih.gov/sites/default/files/documents/health/publications/depression/depression-what-you-need-to-know-2021.pdf",
    "https://www.nimh.nih.gov/sites/default/files/documents/health/publications/anxiety-disorders/anxiety-disorders.pdf",
    "https://afsp.org/wp-content/uploads/2016/01/talked-about-suicide.pdf"
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8448293" # — first national study of US hotline callers
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8485743" # — 5,001 call analysis with demographics
    "https://988lifeline.org/wp-content/uploads/2023/10/988-Suicide-and-Crisis-Lifeline-2022-Annual-Report.pdf"
    #suicide risk assessment protocol
    "https://cssrs.columbia.edu/wp-content/uploads/C-SSRS_Pediatric-SLC_11.14.16.pdf" # — Columbia Suicide Severity Rating Scale
    "https://www.integration.samhsa.gov/clinical-practice/Suicide_Risk_Assessment_Tool.pdf" # — Suicide Risk Assessment Tool
    #domestic violence protocols
    "https://www.thehotline.org/wp-content/uploads/media/2021/03/TheHotline-Safety-Planning-Guide.pdf",
    "https://www.cdc.gov/violenceprevention/pdf/ipv/ipvandsvscreening.pdf"
    # LGBTQ+ crisis resources
    "https://www.thetrevorproject.org/resources/article/facts-about-lgbtq-youth-suicide"
    "https://www.samhsa.gov/sites/default/files/samhsa-lgbtq-behavioral-health.pdf"
    #veteran/first responder protocols
    "https://www.veteranscrisisline.net/media/p1thmnlt/vcl-responder-toolkit.pdf"
    "https://www.samhsa.gov/sites/default/files/practitioner-training-suicide-veterans.pdf"
    # trauma/PTSD grounding techniques
    "https://www.ptsd.va.gov/professional/treat/txessentials/grounding_techniques.asp"
    "https://www.nctsn.org/sites/default/files/resources/fact-sheet/grounding_techniques.pdf"
    # substance abuse protocols
    "https://store.samhsa.gov/sites/default/files/tip35.pdf" # — Motivational Interviewing
    "https://nida.nih.gov/sites/default/files/drugfacts-treatment-approaches.pdf"







]

os.makedirs("docs", exist_ok=True)

for url in sources:
    print(f"Scraping {url}...")
    try:
        result = client.extract(urls=[url])
        content = result["results"][0]["raw_content"]
        
        # Save as text file in docs
        filename = url.replace("https://", "").replace("/", "_") + ".txt"
        with open(f"docs/{filename}", "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved {filename}")
    
    except Exception as e:
        print(f"Failed {url}: {e}")

print("\nDone! All sources saved to docs/")