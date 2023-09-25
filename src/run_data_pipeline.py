from data.odds_scraper import BestFightOddsScraper
from data.ufc_scraper import UFCScraper
from features.feature_extractor import FeatureExtractor
from features.transformer import Transformer


def run_data_pipeline():
    # Pull stats from ufcstats to raw data
    data_scraper = UFCScraper()
    data_scraper.scrape_fights()

    # Do first transformation
    transform = Transformer()
    transform.write_transform()

    # Do feature extraction 
    feat = FeatureExtractor()
    feat.extract()
    feat.write()

    # Do odds scraping and merging 
    odds_scraper = BestFightOddsScraper()
    odds_scraper.run()

if __name__ == '__main__':
    run_data_pipeline()

