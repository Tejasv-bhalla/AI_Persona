"""Resolving "the stock market project" to a repo name before searching Qdrant.

Getting this wrong is worse than not filtering: a wrong repo filter hides the
answer entirely, so ambiguity has to resolve to None.
"""

from rag_persona.nodes.retrieval import extract_repo_filter, repo_tokens

REPOS = [
    "audio-emotion-classification",
    "Credit-Default-Prediction",
    "JHealth",
    "Stock-Market-Prediction",
    "TalentScoutBot",
    "Shramik.ai",
    "AI_Persona",
    "LegalX",
]


class FakeStore:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def repo_names(self) -> list[str]:
        return self.names


STORE = FakeStore(REPOS)


def resolve(text: str) -> str | None:
    return extract_repo_filter(text, STORE)  # type: ignore[arg-type]


def test_a_repo_name_split_into_ordinary_words_still_resolves() -> None:
    assert resolve("tell me about the stock market project") == "Stock-Market-Prediction"
    assert resolve("how does credit default work") == "Credit-Default-Prediction"
    assert resolve("what is the audio emotion classification model") == (
        "audio-emotion-classification"
    )


def test_camel_case_repo_names_resolve_from_their_spoken_form() -> None:
    assert resolve("the talent scout bot") == "TalentScoutBot"
    assert resolve("what about legal x") == "LegalX"


def test_a_transcription_of_a_repo_name_resolves_on_its_decisive_token() -> None:
    """"JHealth" is heard as "jilo health"; "health" alone names only that repo."""
    assert resolve("tell me about jilo health") == "JHealth"
    assert resolve("what is shramik") == "Shramik.ai"


def test_a_token_shared_by_several_repos_is_not_decisive() -> None:
    assert resolve("what prediction models has he built") is None
    assert resolve("tell me about his ai work") is None


def test_a_tie_between_two_repos_leaves_the_search_unfiltered() -> None:
    """"the persona bot" names AI_Persona and TalentScoutBot equally well."""
    assert resolve("the persona bot") is None


def test_a_question_that_mentions_no_project_is_not_filtered() -> None:
    assert resolve("where did he study") is None
    assert resolve("") is None


def test_a_missing_or_empty_index_returns_none_instead_of_raising() -> None:
    assert extract_repo_filter("the stock market project", None) is None
    assert extract_repo_filter("the stock market project", FakeStore([])) is None  # type: ignore[arg-type]


def test_repo_names_tokenize_on_both_delimiters_and_camel_case() -> None:
    assert repo_tokens("Stock-Market-Prediction") == {"stock", "market", "prediction"}
    assert repo_tokens("TalentScoutBot") == {"talent", "scout", "bot"}
    # A one-character token is noise, not a search term.
    assert repo_tokens("LegalX") == {"legal"}
