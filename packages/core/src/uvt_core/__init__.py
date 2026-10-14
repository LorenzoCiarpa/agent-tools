"""Shared, lightweight building blocks for uvt tools.

Import submodules explicitly (``from uvt_core.jobs import JobSpec``); this
package keeps imports cheap so ``--help`` stays fast.
"""

from uvt_core.jobs import package_version

__version__ = package_version("uvt-core")
