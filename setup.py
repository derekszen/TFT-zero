from __future__ import annotations

import sys

from setuptools import Extension, setup

extra_compile_args = ["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3"]

setup(
    ext_modules=[
        Extension(
            "mini_tft.strategic.native._native",
            sources=[
                "src/mini_tft/strategic/native/cpp/bindings.cpp",
                "src/mini_tft/strategic/native/cpp/strategic_mcts.cpp",
                "src/mini_tft/strategic/native/cpp/strategic_rules.cpp",
            ],
            include_dirs=["src/mini_tft/strategic/native/cpp"],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ],
)
