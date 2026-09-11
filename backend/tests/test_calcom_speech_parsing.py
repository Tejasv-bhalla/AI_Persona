"""Speech-transcript parsing for the voice booking flow.

These helpers run on speech-to-text output, which mangles email addresses and
times in specific, repeatable ways. Everything here is a pure function.
"""


from rag_persona.nodes.calcom import (
    detect_slot_selection,
    extract_email,
    extract_name,
    format_ordinal,
    to_spoken_slot,
)

# Mon 10:00, Tue 10:30, Tue 14:00, Wed 09:00 — fixed, never derived from now().
SLOTS = [
    "2026-09-07T10:00:00Z",
    "2026-09-08T10:30:00Z",
    "2026-09-08T14:00:00Z",
    "2026-09-09T09:00:00Z",
]


# --- extract_email ---------------------------------------------------------


def test_spoken_dot_and_at_become_punctuation() -> None:
    assert extract_email("john dot carter at gmail dot com") == "john.carter@gmail.com"


def test_at_the_rate_is_read_as_an_at_sign() -> None:
    """Indian-English speakers say "at the rate"; Deepgram transcribes it literally."""
    assert extract_email("my email is john at the rate gmail dot com") == "john@gmail.com"
    assert extract_email("john at the rate of gmail dot com") == "john@gmail.com"


def test_third_egg_mail_is_recovered_as_gmail() -> None:
    """A real transcription failure of "at gmail" that would otherwise lose the domain."""
    assert extract_email("tejasv at third egg mail dot com") == "tejasv@gmail.com"


def test_bracketed_at_and_dot_forms_are_understood() -> None:
    assert extract_email("john [at] gmail [dot] com") == "john@gmail.com"


def test_trailing_sentence_punctuation_is_stripped() -> None:
    assert extract_email("My email is bob@example.com.") == "bob@example.com"
    assert extract_email("is it sara@example.com?") == "sara@example.com"


def test_an_utterance_with_no_email_returns_none_rather_than_a_garbage_match() -> None:
    assert extract_email("not an email") is None
    assert extract_email("my name is John Carter") is None
    assert extract_email("hello are you still there") is None
    assert extract_email("") is None


def test_a_spelled_out_address_keeps_every_letter() -> None:
    """Regression: the first match won on un-collapsed text, yielding only the last letter.

    This is the path callers take after the "could you spell it out" re-prompt.
    """
    assert extract_email("j o h n at gmail dot com") == "john@gmail.com"


def test_multi_part_domains_are_not_truncated() -> None:
    """Regression: the domain allowed one dot, silently truncating before the TLD.

    EmailStr accepts the truncation, so the caller was told an invite had been sent
    to an address that does not exist. Covers multi-part TLDs and subdomains.
    """
    assert extract_email("sarah at the rate of yahoo dot co dot uk") == "sarah@yahoo.co.uk"
    assert extract_email("jane dot doe at sub dot example dot org") == "jane.doe@sub.example.org"
    assert extract_email("my email is bob at mail dot company dot com") == "bob@mail.company.com"


# --- extract_name ----------------------------------------------------------


def test_leading_filler_words_are_dropped_from_a_name() -> None:
    assert extract_name("uhm, my name is John Carter") == "John Carter"
    assert extract_name("yeah this is Mike") == "Mike"


def test_spoken_lead_ins_are_stripped_from_a_name() -> None:
    assert extract_name("this is Sarah Connor") == "Sarah Connor"
    assert extract_name("I am bob smith") == "Bob Smith"
    assert extract_name("sure, my name is Ana") == "Ana"


def test_a_bare_name_is_returned_title_cased() -> None:
    assert extract_name("priya") == "Priya"


# --- detect_slot_selection -------------------------------------------------


def test_ordinal_references_pick_the_nth_offered_slot() -> None:
    assert detect_slot_selection("the third one", SLOTS) == SLOTS[2]
    assert detect_slot_selection("the second one please", SLOTS) == SLOTS[1]


def test_option_and_number_phrasings_pick_the_nth_offered_slot() -> None:
    assert detect_slot_selection("option two", SLOTS) == SLOTS[1]
    assert detect_slot_selection("number one", SLOTS) == SLOTS[0]


def test_a_weekday_with_a_time_picks_that_exact_slot() -> None:
    assert detect_slot_selection("Tuesday at 10:30", SLOTS) == SLOTS[1]
    assert detect_slot_selection("monday at 10", SLOTS) == SLOTS[0]
    assert detect_slot_selection("let's do tuesday at two", SLOTS) == SLOTS[2]


def test_half_past_is_not_confused_with_the_top_of_the_hour() -> None:
    """Two Tuesday slots exist; "ten" alone must not silently resolve to the 10:30 one."""
    assert detect_slot_selection("tuesday at ten thirty", SLOTS) == SLOTS[1]
    assert detect_slot_selection("tuesday at ten", SLOTS) is None


def test_a_bare_weekday_resolves_only_when_one_slot_falls_on_it() -> None:
    assert detect_slot_selection("wednesday works", SLOTS) == SLOTS[3]
    # Tuesday holds two slots, so the day alone is not a choice.
    assert detect_slot_selection("tuesday works", SLOTS) is None


def test_an_utterance_naming_no_slot_returns_none() -> None:
    assert detect_slot_selection("whatever works for you", SLOTS) is None
    assert detect_slot_selection("no that doesn't work", SLOTS) is None
    assert detect_slot_selection("the first one", []) is None


# --- to_spoken_slot / format_ordinal ---------------------------------------


def test_teens_take_th_not_st_nd_rd() -> None:
    """The classic off-by-one: 11/12/13 must not follow the 1/2/3 suffix rule."""
    assert format_ordinal(11) == "11th"
    assert format_ordinal(12) == "12th"
    assert format_ordinal(13) == "13th"


def test_ordinary_days_take_the_expected_suffix() -> None:
    assert format_ordinal(1) == "1st"
    assert format_ordinal(2) == "2nd"
    assert format_ordinal(3) == "3rd"
    assert format_ordinal(21) == "21st"
    assert format_ordinal(4) == "4th"


def test_on_the_hour_slots_are_spoken_without_double_zero_minutes() -> None:
    assert to_spoken_slot("2026-09-08T09:00:00Z") == "Tuesday, September 8th at 9 AM"
    assert to_spoken_slot("2026-09-11T13:00:00Z") == "Friday, September 11th at 1 PM"


def test_off_the_hour_slots_keep_their_minutes() -> None:
    assert to_spoken_slot("2026-09-08T10:30:00Z") == "Tuesday, September 8th at 10:30 AM"


def test_an_unparseable_timestamp_is_spoken_back_verbatim() -> None:
    assert to_spoken_slot("not a timestamp") == "not a timestamp"
