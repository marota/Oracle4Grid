import setuptools
from setuptools import setup

pkgs = {
    "required": [
        "numba==0.53.1",
        "grid2op>1.5.1",
        "pandas==1.2.4",
        "psutil==5.7.2",
        "matplotlib==3.3.2",
        "pybind11==2.5.0",
        "ipykernel==5.3.4",
        "ipywidgets==7.5.1",
        "numpy==1.19.3",
        "pytest==6.2.2",
        # We don't need these
        "scipy<=1.6.0"
    ]
}

setup(name='Oracle4Grid',
      version='1.0.3',
      description='Oracle agent that finds the best course of actions aposteriori, within a given perimeter of actions',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='ML powergrid optmization RL power-systems',
      author='Nicolas Megel, Mario Jothy',
      author_email='nico.megel@gmail.com; mariojothy@gmail.com',
      url="https://github.com/marota/Oracle4Grid/",
      license='Mozilla Public License 2.0 (MPL 2.0)',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={'oracle': ["oracle4grid/ressources/config.ini",
                                  "oracle4grid/ressources/rte_case14_realistic/test_unitary_actions.json"]},
      install_requires=pkgs["required"],
      zip_safe=False,
      entry_points={'console_scripts': ['oracle4grid=oracle4grid.main:main']}
)