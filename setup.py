import setuptools
from setuptools import setup

pkgs = {
    "required": [
        "numba==0.49.1",
        # "grid2op==1.1.X", # Expecting release - for now grid2op should be installed from https://github.com/mjothy/Grid2Op.git@mj-devs-pr
        "pandas==1.1.3",
        "psutil==5.7.2",
        "matplotlib==3.3.2",
        "pybind11==2.5.0",
        "ipykernel==5.3.4",
        "ipywidgets==7.5.1",
        "numpy==1.19.3",
        "pytest==6.2.2"
    ]
}

setup(name='Oracle4Grid',
      version='0.0.1',
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
      author='Nicolas Megel',
      author_email='nico.megel@gmail.com',
      url="https://github.com/marota/Oracle4Grid/",
      license='Mozilla Public License 2.0 (MPL 2.0)',
      packages=setuptools.find_packages(),
      #extras_require=pkgs["extras"],
      include_package_data=True,
      package_data={'alphaDeesp': ["oracle4grid/ressources/config.ini",
                                  "oracle4grid/ressources/rte_case14_realistic/test_unitary_actions.json"]},
      install_requires=pkgs["required"],
      zip_safe=False,
      entry_points={'console_scripts': ['oracle4grid=main:main']}
)