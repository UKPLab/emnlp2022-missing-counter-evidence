import re
from typing import List, Dict


class RegexpHolder:
    """
    Contains regular expressions used to infer whether we consider an evidence snippet as leaked based on the content.
    """

    @staticmethod
    def get() -> List:
        """
        Return the used regular expressions.
        """

        regular_expressions = [
            re.compile(exp) for exp in [
                r'^false:', r'politifact', r'snopes', r'^debunk',
                r'real story behind', r'\bfake\b', r'\bhoax\b', r'\bfalsely\b',
                r'\brumors?\b', r'\bmyths?\b', r'\bnot real news\b', 'r\bunfounded\b', r'fact[ -]check'
            ]
        ]

        return regular_expressions


class LeakedUrlHolder:
    """
    Contains the URL phrases used to determine whether a snippet URL points to a fact-checking article.
    """

    @staticmethod
    def get() -> Dict[str, Dict]:
        """
        Get all URL phrases that may point to a fact-checking article. The result is a dictionary. Each key
        represents a URL phrase. The field "only_fc" indicates whether sources behind this URL point always to
        fact-checking articles (True), or whether the outlet behind the URL may also publish non-fact-checking
        articles (False).

        We use the stricter version (True) in the paper.
        """

        # These URLs publish fact-checking articles among other news/content.
        can_be_fc_article: List[str] = [
            "abc.net.au/news/", "boomlive.in", "huffingtonpost.ca", "blogs.mprnews.org",
            "observatory.journalism.wisc.edu", "pandora.nla.gov.au", "theferret.scot", "thejournal.ie",
            "verafiles.org", "voiceofsandiego.org"
        ]

        # These URLs only focus on fact-checking articles / only point to fact-checking articles.
        is_fc_article: List[str] = [
            "africacheck.org/reports", "checkyourfact.com", "climatefeedback.org/claimreview", "factscan.ca",
            "factly.in", "factcheckni.org", "factcheck.org", "gossipcop.com", "hoax-slayer.net", "politifact.com.au",
            "politifact.com", "pesacheck.org", "radionz.co.nz/programmes/election17-fact-or-fiction", "snopes.com",
            "truthorfiction.com", "washingtonpost.com/news/fact-checker", "fullfact.org",
            "healthfeedback.org/claimreview", "factcheck.afp.com", "hoax-alert.leadstories.com"
        ]

        return {
            url: {"only_fc": only_fc}
            for only_fc, urls in [
                (False, can_be_fc_article), (True, is_fc_article)
            ]
            for url in urls
        }


class FCLeakedClassifier:
    """
    Uses patterns to classify whether the content or the URL of a snippet is considered leaked.
    """

    def __init__(self, url_dict: Dict[str, Dict], regexp: List, strict: bool = True):
        """
        :param url_dict: Dictionary with leaked URL phrases (as provided by LeakedUrlHolder.get)
        :param regexp:  List of regular expressions (as provided by RegexpHolder.get)
        :param strict: Boolean value to indicate whether only URL phrases should be considered that strictly point to
        fact-checking articles (default: True)
        """

        if strict:
            self.urls = [k for k in url_dict if url_dict[k]['only_fc']]
        else:
            self.urls = list(url_dict.keys())

        self.leakage_regular_expressions = regexp

    def is_fc_url(self, url: str) -> bool:
        """
        Classify whether a URL points to a fact-checking article (True) or not (False)
        :param url: The URL to check.
        """

        for fc_url in self.urls:
            if fc_url in url:
                return True
        return False

    def is_fc_statement(self, snippet: Dict[str, str]) -> bool:
        """
        Classify whether the snippet (title or text) contain phrases indicating it is leaked (True) or not (False).
        :param snippet: Dictionary containing title and text of an evidence snippet.
        :return:
        """
        snippet_title = snippet['snippet_title'] if snippet['snippet_title'] is not None else ''
        snippet_text = snippet['snippet_text'] if snippet['snippet_text'] is not None else ''

        for text in [snippet_title, snippet_text]:
            text = text.lower()
            for pattern in self.leakage_regular_expressions:
                if re.search(pattern, text):
                    return True
        return False

