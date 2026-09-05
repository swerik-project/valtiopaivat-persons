"""
Triangulate person data from different sources.
"""
from glob import glob
from trainerlog import get_logger
import json
import os
import polars as pl
import re
import unittest




LOGGER = get_logger(name=f"{__file__.split("/")[-1][:-3]}")




class RegistryBiobookCorrespondence(unittest.TestCase):
    """
    Compares person data scraped from biobooks with person data in the person registry and wikidata.
    """
    @classmethod
    def setUpClass(cls):
        with open("data/person-registry.json", 'r') as regin:
            cls.registry = json.load(regin)
        cls.bb_df = pl.read_csv("test/data/diet_individuals.tsv", separator="\t")
        cls.wid_df = pl.read_csv("test/data/scraped-wikidata-ids.csv")
        cls.write_diagnostics = os.environ.get("WRITE_TEST_DIAGNOSTICS")
        cls.unmatched_reg_entries = set()
        cls.unmatched_BB_entries = set()
        cls.unmatched_wd_entries = set()
        cls.unmatched_wp_entries = set()


    @classmethod
    def tearDownClass(cls):
        if cls.write_diagnostics:
            LOGGER.info("Test finished. Writing diagnostic files")
            os.makedirs("test/results", exist_ok=True)
            unm_reg = pl.DataFrame(sorted(cls.unmatched_reg_entries),
                                   schema=["person_id", "name"],
                                   orient="row")
            unm_bb = pl.DataFrame([(name,) for name in sorted(cls.unmatched_BB_entries)],
                                  schema=["name"],
                                  orient="row")
            unm_wp = pl.DataFrame([x for x in cls.unmatched_wp_entries],
                                  schema=["wiki_id"],
                                  orient="row")
            unm_wd = pl.DataFrame([x for x in cls.unmatched_wd_entries],
                                  schema=["wiki_id"],
                                  orient="row")
            unm_reg.sort("name").write_csv("test/results/unmatched-person-register-entries.csv", separator=";")
            unm_bb.write_csv("test/results/unmatched-biobook-entries.csv", separator=";")
            unm_wd.write_csv("test/results/unmatched-wikidata-wdids.csv", separator=";")
            unm_wp.write_csv("test/results/unmatched-wikipediaa-wdids.csv", separator=";")
            LOGGER.info(" --> DONE.")
        else:
            LOGGER.info("Test finished. Not writing any diagnostics.")


    def test_compare_counts(self):
        """
        Compare the raw number of entries in the two sources
        """
        LOGGER.train("====================")
        LOGGER.info("Compare counts test")
        LOGGER.info(f"Entries from BioBooks: {self.bb_df.height}")
        LOGGER.info(f"Entries in Person Registry: {len(self.registry)}")
        LOGGER.train("====================")


    def test_union_of_set(self):
        """
        Check how much the two datasources overlap.
        """
        def _tokens(name):
            return re.findall(r"\w+", str(name).lower())

        def _given_names_match(reg_tokens, bb_tokens):
            if not reg_tokens or not bb_tokens:
                return False
            if len(reg_tokens) > len(bb_tokens):
                reg_tokens, bb_tokens = bb_tokens, reg_tokens
            for reg_token, bb_token in zip(reg_tokens, bb_tokens):
                shorter, longer = sorted([reg_token, bb_token], key=len)
                if len(shorter) <= 2:
                    if not longer.startswith(shorter):
                        return False
                elif shorter != longer:
                    return False
            return True

        def _reg_given_tokens(reg_name, bb_surname):
            reg_tokens = _tokens(reg_name)
            surname_tokens = _tokens(bb_surname)
            if len(reg_tokens) <= len(surname_tokens):
                return []
            if reg_tokens[-len(surname_tokens):] != surname_tokens:
                return []
            return reg_tokens[:-len(surname_tokens)]

        def _name_matches(reg_name, bb_names, bb_surname):
            return _given_names_match(
                _reg_given_tokens(reg_name, bb_surname),
                _tokens(bb_names),
            )

        def _has_short_name(tokens):
            return any(len(token) <= 2 for token in tokens)

        def _is_ambiguous_short_match(reg_given, bb_names, bb_surname):
            if not (_has_short_name(reg_given) or _has_short_name(_tokens(bb_names))):
                return False

            bb_matches = [
                row
                for row in bb_rows
                if _tokens(row[1]) == _tokens(bb_surname)
                and _given_names_match(_tokens(row[0]), _tokens(bb_names))
            ]
            reg_matches = {
                reg_id
                for reg_id, reg_entry in self.registry.items()
                for reg_name in reg_entry.get("names", [])
                if _given_names_match(
                    _reg_given_tokens(reg_name, bb_surname),
                    reg_given,
                )
            }
            return len(bb_matches) > 1 or len(reg_matches) > 1

        def _reg_entry_matches(reg_entry, bb_names, bb_surname):
            reg_given = [
                _reg_given_tokens(reg_name, bb_surname)
                for reg_name in reg_entry.get("names", [])
                if _name_matches(reg_name, bb_names, bb_surname)
            ]
            return bool(reg_given) and not _is_ambiguous_short_match(
                reg_given[0],
                bb_names,
                bb_surname,
            )

        def _single_registry_match(bb_names, bb_surname):
            candidates = {
                person_id
                for person_id, reg_entry in self.registry.items()
                if _reg_entry_matches(reg_entry, bb_names, bb_surname)
            }
            return next(iter(candidates)) if len(candidates) == 1 else None


        UNION_COUNT = 0
        cls = self.__class__

        bb_rows = [
            (row["names"], row["surname"])
            for row in self.bb_df.iter_rows(named=True)
        ]
        bb_registry_matches = [
            _single_registry_match(bb_names, bb_surname)
            for bb_names, bb_surname in bb_rows
        ]

        for person_id, reg_entry in self.registry.items():
            if person_id in bb_registry_matches:
                UNION_COUNT += 1
            else:
                cls.unmatched_reg_entries.add((person_id, reg_entry["names"][0]))

        for (bb_names, bb_surname), person_id in zip(bb_rows, bb_registry_matches):
            if person_id is None:
                cls.unmatched_BB_entries.add(f"{bb_names} {bb_surname}")

        LOGGER.train("====================")
        LOGGER.info("Compare union test")
        LOGGER.info(f"Matched across sources: {UNION_COUNT}")
        LOGGER.info(f"Unmatched BioBook entries: {len(cls.unmatched_BB_entries)}")
        LOGGER.info(f"Unmatched Registry entries: {len(cls.unmatched_reg_entries)}")
        LOGGER.train("====================")


    def test_compare_wiki_ids(self):
        """
        Compare wiki IDs scraped from wikidata and those via wikipedia
        """
        cls = self.__class__
        from_wikipedia = set()
        for f in glob("data/sources/wikipedia/*.csv"):
            df = pl.read_csv(f, separator=";")
            wids = set(df["wikidata_id"].drop_nulls()) - {""}
            for wid in wids:
                from_wikipedia.add(wid)
        LOGGER.train("====================")
        LOGGER.info("Compare wiki_ids")
        LOGGER.info(f"Wiki_ids via wikidata: {self.wid_df.height}")
        LOGGER.info(f"Wiki_ids via wikipedia: {len(from_wikipedia)}")
        LOGGER.train("====================")

        from_wikidata = set(self.wid_df["wiki_id"])
        cls.unmatched_wp_entries = from_wikipedia-from_wikidata
        cls.unmatched_wd_entries = from_wikidata-from_wikipedia

        LOGGER.train("====================")
        LOGGER.info("Wiki ID unions")
        LOGGER.info(f"Matched across sources: {len(from_wikipedia & from_wikidata)}")
        LOGGER.info(f"Unmatched from wikipedia: {len(cls.unmatched_wp_entries)}")
        LOGGER.info(f"Unmatched from wikidata: {len(cls.unmatched_wd_entries)}")
        LOGGER.train("====================")




if __name__ == '__main__':
    unittest.main()
