# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "TC"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.tc_obj_root, "test")
config.tc_tools_dir = os.path.join(config.tc_obj_root, "bin")
config.substitutions.append(("%tc_src", config.tc_src_root))

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.tc_tools_dir, config.llvm_tools_dir]
tools = ["tc-opt", "tc-compile", "FileCheck"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
