"""
=========================================================================
OpuIsa_test.py
=========================================================================
Includes test cases for OpuIsa
"""
import random
import pytest
import pdb

#from ProcFL import ProcFL
from pymtl3 import *
#from pymtl3.stdlib.test_utils import run_sim

from test import (
    inst_add,
)
from test.harness import TestHarness, asm_test

random.seed(0xdeadbeef)

#-------------------------------------------------------------------------
# OpuIsa_Tests
#-------------------------------------------------------------------------
# We group all our test cases into a class
class Test_OpuIsa:
  # [setup_class] will be called by pytest before running all the tests in
  # the test class. Here we specify the type of the processor that is used
  # in all test cases. We can easily reuse all these test cases in simply
  # by creating a new test class that inherits from this class and
  # overwrite the setup_class to provide a different processor type.
  @classmethod
  def setup_class( cls ):
    pass
    #cls.ProcType = ProcFL

  # [run_sim] is a helper function in the test suite that creates a
  # simulator and runs test. We can overwrite this function when
  # inheriting from the test class to apply different passes to the DUT.
  def run_sim(self, th, name, gen_test):
    #pdb.set_trace()
    #th.elaborate()
    # Assemble the program
    kname = "kernel_{}".format(name)
    th.assemble(kname, gen_test())
    return th.run(kname)

    # Load the program into memory
    #th.load(mem_image )


  #-----------------------------------------------------------------------
  # add
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_add.gen_empty_test ) ,
    asm_test( inst_add.gen_add_basic_test ) ,
  ])
  def test_add(self, name, test):
    th = TestHarness()
    result = self.run_sim(th, name, test)
    assert result == True


