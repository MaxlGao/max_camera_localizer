from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'max_camera_localizer'

data_files = []
for dirpath, dirnames, filenames in os.walk(os.path.join('pusher_data')):
    for f in filenames:
        full_path = os.path.join(dirpath, f)
        install_path = os.path.join('share', package_name, dirpath)
        data_files.append((install_path, [full_path]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/STL', glob('STL/*.STL')),
        *data_files
    ],
    install_requires=['setuptools', 'cv2', 'scipy'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='max@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localize = max_camera_localizer.merged_localization:main',
            'allocate = max_camera_localizer.control_allocation_node:main'
        ],
    },
)
