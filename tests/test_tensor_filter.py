import fnmatch
import random
import unittest
from unittest import mock

from instanttensor import safe_open
import instanttensor._impl as impl


class TensorFilterTest(unittest.TestCase):
    def test_default_is_glob(self):
        matches = impl.compile_name_matcher(["model.layers.*.weight"])
        self.assertTrue(matches("model.layers.0.weight"))
        self.assertTrue(matches("model.layers.0.attn.weight"))
        self.assertFalse(matches("modelXlayersX0Xweight"))

    def test_glob_metacharacters(self):
        matches = impl.compile_name_matcher(["layer?.weight[012]"])
        self.assertTrue(matches("layer0.weight2"))
        self.assertFalse(matches("layer10.weight2"))
        self.assertFalse(matches("layer0.weight3"))
        self.assertTrue(impl.compile_name_matcher(["weight[[]0]"])("weight[0]"))

    def test_glob_literals_skip_regex_compilation(self):
        names = ["model.layers.0.weight", "weight]", "weight!", "weight-",
                 "weight{0}", "weight(0)", "weight+", "weight^$", "weight|0", r"weight\0", ""]
        with mock.patch.object(impl.re, "compile", wraps=impl.re.compile) as compile_pattern:
            matches = impl.compile_name_matcher(names)
            for name in names:
                self.assertIs(matches(name), True)
            self.assertIs(matches("modelXlayersX0Xweight"), False)
            self.assertIs(matches("Weight]"), False)
            compile_pattern.assert_not_called()

    def test_glob_mixed_patterns_compile_only_wildcards(self):
        patterns = ["literal.weight", "literal.weight", "layer*.bias", "layer?.weight",
                    "weight[012]", "broken["]
        with mock.patch.object(impl.re, "compile", wraps=impl.re.compile) as compile_pattern:
            matches = impl.compile_name_matcher(patterns)
            self.assertEqual(compile_pattern.call_count, 4)
            for name in ("literal.weight", "layer0.attn.bias", "layer1.weight", "weight0", "broken["):
                self.assertIs(matches(name), True)
            for name in ("literalXweight", "layer10.weight", "weight3", "absent"):
                self.assertIs(matches(name), False)
            self.assertEqual(compile_pattern.call_count, 4)

    def test_regex_does_not_use_glob_literal_fast_path(self):
        with mock.patch.object(impl.re, "compile", wraps=impl.re.compile) as compile_pattern:
            matches = impl.compile_name_matcher(["weight", "model.layers.0.weight"], "regex")
            self.assertEqual(compile_pattern.call_count, 2)
            self.assertTrue(matches("modelXlayersX0Xweight"))
            self.assertTrue(matches("weight"))

    def test_glob_fast_path_matches_fnmatchcase(self):
        rng = random.Random(29)
        alphabet = r"ab01.*?[]!^-\(){}|+$"
        for trial in range(50):
            patterns = ["literal.weight", "weight]", r"weight\0"] + [
                "".join(rng.choices(alphabet, k=rng.randrange(1, 12))) for _ in range(6)
            ]
            names = patterns + ["".join(rng.choices(alphabet, k=rng.randrange(0, 12)))
                                for _ in range(30)]
            matches = impl.compile_name_matcher(patterns)
            for name in names:
                with self.subTest(trial=trial, name=name):
                    self.assertEqual(matches(name), any(fnmatch.fnmatchcase(name, p) for p in patterns))

    def test_exact_treats_special_characters_literally(self):
        names = ["weight[0]", "weight*", "weight?", "broken[", "(foo"]
        matches = impl.compile_name_matcher(names, "exact")
        for name in names:
            self.assertTrue(matches(name))
        self.assertFalse(matches("weight0"))
        self.assertFalse(matches("weight_anything"))

    def test_regex_fullmatch(self):
        matches = impl.compile_name_matcher([r"model\.layers\.\d+\.weight"], "regex")
        self.assertTrue(matches("model.layers.12.weight"))
        self.assertFalse(matches("prefix.model.layers.12.weight"))
        self.assertFalse(matches("model.layers.12.weight.suffix"))

    def test_case_sensitive_by_default(self):
        for mode in ("exact", "glob", "regex"):
            with self.subTest(mode=mode):
                matches = impl.compile_name_matcher(["weight"], mode)
                self.assertTrue(matches("weight"))
                self.assertFalse(matches("Weight"))

    def test_none_and_empty_list(self):
        for mode in ("exact", "glob", "regex"):
            with self.subTest(mode=mode):
                self.assertTrue(impl.compile_name_matcher(None, mode)("anything"))
                self.assertFalse(impl.compile_name_matcher([], mode)("anything"))

    def test_patterns_are_alternatives_and_return_bool(self):
        for mode in ("exact", "glob", "regex"):
            with self.subTest(mode=mode):
                matches = impl.compile_name_matcher(["first", "second"], mode)
                self.assertIs(matches("first"), True)
                self.assertIs(matches("second"), True)
                self.assertIs(matches("third"), False)

    def test_invalid_mode(self):
        for mode in ("auto", "GLOB", "", None, [], 1):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "tensor_name_match_mode"):
                impl.compile_name_matcher(None, mode)

    def test_requires_string_list(self):
        for filters in ("weight", [1], [None], ("weight",), {"weight"}, lambda name: True):
            with self.subTest(filters=filters), self.assertRaisesRegex(TypeError, "list of strings"):
                impl.compile_name_matcher(filters)

    def test_invalid_regex_does_not_affect_other_modes(self):
        with self.assertRaisesRegex(ValueError, "Invalid regex pattern"):
            impl.compile_name_matcher(["broken["], "regex")
        for mode in ("exact", "glob"):
            self.assertTrue(impl.compile_name_matcher(["broken["], mode)("broken["))

    def test_public_api_validates_before_reading_metadata(self):
        with mock.patch.object(safe_open, "_read_metadata") as read:
            with self.assertRaisesRegex(ValueError, "tensor_name_match_mode"):
                safe_open([], "pt", "cuda:0", tensor_name_match_mode="auto")
            with self.assertRaisesRegex(ValueError, "Invalid regex"):
                safe_open([], "pt", "cuda:0", tensor_filter=["["], tensor_name_match_mode="regex")
            read.assert_not_called()

    def test_selection_uses_only_the_explicit_mode(self):
        names = ["weight[0]", "weight0", "other.weight0", "Weight0", "weight1"]
        metadata = {
            name: {"dtype": "U8", "shape": [4], "data_offsets": [4 * i, 4 * (i + 1)]}
            for i, name in enumerate(names)
        }
        cases = [
            ({"tensor_filter": ["weight[0]"]}, ["weight0"]),
            ({"tensor_filter": ["weight[0]"], "tensor_name_match_mode": "exact"}, ["weight[0]"]),
            ({"tensor_filter": ["weight[0]"], "tensor_name_match_mode": "regex"}, ["weight0"]),
            ({"tensor_filter": ["weight*"]}, ["weight[0]", "weight0", "weight1"]),
            ({"tensor_filter": None}, names),
            ({"tensor_filter": []}, []),
        ]
        for kwargs, selected in cases:
            with self.subTest(kwargs=kwargs), \
                 mock.patch.object(safe_open, "_determine_io_params"), \
                 mock.patch.object(safe_open, "_finalize_buffer_size"), \
                 mock.patch.object(safe_open, "_read_metadata", return_value=[(None, metadata, 64)]):
                loader = safe_open("mock.safetensors", "pt", "cuda:0", load_now=False, **kwargs)
                self.assertEqual(loader.keys(), selected)
                self.assertEqual(loader.total_tensor_size, 4 * len(selected))
                self.assertEqual(loader.tensor_keep, [name in selected for name in names])
                self.assertTrue(all(type(keep) is bool for keep in loader.tensor_keep))


if __name__ == "__main__":
    unittest.main()
