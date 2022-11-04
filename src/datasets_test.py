import textwrap

from absl.testing import absltest

from datasets import ConllXDatasetPos, _make_fields
import test_utils


class MyTestCase(absltest.TestCase):
    def test_read_file(self):
        test_utils.ensure_flag("test_tmpdir")

        TEMPORARY_CONTENT = textwrap.dedent(
            """\
        1	No	_	RB	RB	_	7	discourse	_	_
        2	,	_	,	,	_	7	punct	_	_
        3	it	_	PRP	PRP	_	7	nsubj	_	_
        4	was	_	VBD	VBD	_	7	cop	_	_
        5	n't	_	RB	RB	_	7	neg	_	_
        6	Black	_	NNP	NNP	_	7	nn	_	_
        7	Monday	_	NNP	NNP	_	0	root	_	_
        8	.	_	.	.	_	7	punct	_	_
        
        1	But	_	CC	CC	_	33	cc	_	_
        """
        )
        file = self.create_tempfile(content=TEMPORARY_CONTENT)
        dataset = ConllXDatasetPos(file, _make_fields())

        self.assertLen(dataset, 2)
        vars_ = [vars(x) for x in dataset]
        self.assertDictEqual(vars_[1], {"word": ["But"], "pos": ["CC"]})


if __name__ == "__main__":
    absltest.main()
