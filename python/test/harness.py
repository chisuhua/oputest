"""
=========================================================================
harness.py
=========================================================================
Includes a test harness that composes a processor, src/sink, and test
memory, and a run_test function.
"""

import struct
import os
import jinja2
import subprocess
import pdb

path = os.path.dirname(os.path.abspath(__file__))
host_execute = os.path.join(path, "../../cuda_samples/smoke/vectorCopy/vectorCopy")
gpgpusim_config = os.path.join(path, "../../gpgpusim.config")
env = jinja2.Environment(loader=jinja2.FileSystemLoader(path))
cwd = os.getcwd()
kernel_name = "_Z10vectorCopyPKiPii"
tpl = env.get_template("{}.tpl".format(kernel_name))

#from examples.ex03_proc.NullXcel import NullXcelRTL
#from examples.ex03_proc.tinyrv0_encoding import assemble
#from pymtl3 import *
#from pymtl3.stdlib.mem.MagicMemoryCL import MagicMemoryCL, mk_mem_msg
#from pymtl3.stdlib.connects import connect_pairs
#from pymtl3.stdlib.test_utils import TestSinkCL, TestSrcCL

#=========================================================================
# TestHarness
#=========================================================================
# Use this with pytest parameterize so that the name of the function that
# generates the assembly test ends up as part of the actual test case
# name. Here is an example:
#
#  @pytest.mark.parametrize( "name,gen_test", [
#    asm_test( gen_basic_test  ),
#    asm_test( gen_bypass_test ),
#    asm_test( gen_value_test  ),
#  ])
#  def test( name, gen_test ):
#    run_test( ProcFL, gen_test )
#

def asm_test( func ):
  name = func.__name__
  if name.startswith("gen_"):
    name = name[4:]
  if name.endswith("_test"):
    name = name[:-5]

  return (name,func)

#=========================================================================
# TestHarness
#=========================================================================
class TestHarness:
  #-----------------------------------------------------------------------
  # assemble
  #-----------------------------------------------------------------------
  def assemble(self, kname, test_code):
    #pdb.set_trace()
    tpl_out = tpl.render(test_code=test_code, kname = kname)
    with open(os.path.join(cwd, "{}".format(kname)), 'w') as f:
      f.writelines(tpl_out)
      f.close()
    pass

  #-----------------------------------------------------------------------
  # run
  #-----------------------------------------------------------------------
  def run(self, kname ):
    #env = os.environ
    #env["PREDEFINE_COASM"]=kname;
    setting = "gpgpusim.config"
    if not os.path.isfile(setting):
        subprocess.run(['cp', gpgpusim_config, setting])
    with open("run_{}.sh".format(kname), 'w') as f:
      f.writelines("PREDEFINE_COASM={} {}".format(kname, host_execute))
    #pdb.set_trace()
    #subprocess.run(host_execute, env=env)
    test_cmd = os.path.join(cwd, "run_{}.sh".format(kname))
    os.system("chmod +x {}".format(test_cmd))
    test_pass = False
    #pdb.set_trace()
    with os.popen(test_cmd, 'r') as p:
        r = p.read()
        if r.find("PASSED") > 0:
            #pdb.set_trace()
            test_pass = True
        #test_pass = True
    return test_pass
    #pdb.set_trace()
    #with open("{}.log".format(kname), 'w') as f:
    #    subprocess.Popen(['sh', run_test], stdout=f)
    #pass
    # Iterate over the sections
    #sections = mem_image.get_sections()

  #-----------------------------------------------------------------------
  # done
  #-----------------------------------------------------------------------

  def done():
    pass
    #return s.src.done() and s.sink.done()

  #-----------------------------------------------------------------------
  # line_trace
  #-----------------------------------------------------------------------

  def line_trace():
    pass
    #return s.src.line_trace()  + " > " + \
    #       s.proc.line_trace() + " > " + \
    #       s.sink.line_trace()
