project=pyjava
version=$(grep -oP '__version__ = "\K[0-9]+\.[0-9]+\.[0-9]+' pyjava/version.py | cut -d '"' -f 1)
echo "Version: $version"

echo "Cleaning dist..."
rm -rf ./dist/*

echo "Uninstall Pyjava dist..."
pip uninstall -y ${project}

echo "Building Pyjava dist..."
python setup.py sdist bdist_wheel
cd ./dist/

echo "Installing Pyjava dist..."
pip install ${project}-${version}-py3-none-any.whl && cd -


export MODE=${MODE:-"dev"}
if [[ ${MODE} == "release" ]];then
 git tag v${version}
 git push gitee v${version}
#  git push origin v${version}
 echo "Uploading Pyjava dist..."
 twine upload dist/*
fi
