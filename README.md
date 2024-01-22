# template-python
Web service template in Python for reuse.

## Installation
1. If you don't have `Poetry` installed run:

```bash
pip install poetry
```

2. Install dependencies:

```bash
poetry config virtualenvs.in-project true
poetry install --no-root --with dev,test
```

3. Install `pre-commit` hooks:

```bash
poetry run pre-commit install
```

4. Launch of the project:

```bash
poetry run uvicorn app.main:app [--reload]
```

5. Running tests:

```bash
poetry run pytest
```

6. Building package:

```bash
poetry build
```

## Docker
The docker image is automatically built in GitHub Actions after pushing code/tag/pr.

You can build a docker image and create a container manually:

```bash
docker build . -t <image-name>:<image-tag>
docker run <image-name>:<image-tag>
```

https://docs.docker.com/

## Release
The release is automatically built in GitHub Actions and saved to branch gh-pages after pushing code/tag/pr.

Create the branch gh-pages and use it as a GitHub page https://pages.github.com/.  
The releases will be available at `https://github.com/<workspace>/<project>/releases/download/<app>-<version>/<app>-<version>.tgz`.  
The index will be available at `https://<workspace>.github.io/<project>/index.yaml`.  
You can use URL `https://<workspace>.github.io/<project>/` on https://artifacthub.io/.

To create a release, add a tag in GIT with the format a.a.a, where 'a' is an integer.
The release version for branches, pull requests, and other tags will be generated based on the last tag of the form a.a.a.

The Helm chart version, and Docker images versions will be automatically generated from this version in GitHub Actions.

You can build release manually:

```bash
helm package charts/<chart-name>
```

## Deploy
The release is automatically deployed to Kubernetes cluster in GitHub Actions.

You can deploy it manually:

```bash
helm repo add <repo-name> https://<workspace>.github.io/<project>/
helm repo update <repo-name>
helm upgrade --install <release-name> <repo-name>/<chart-name>
```

https://helm.sh/ru/docs/

## GitHub Actions
GitHub Actions run tests, build and push a Docker image, creat and push a Helm chart release, deploy the project to Kubernetes cluster.

Setup secrets at `https://github.com/<workspace>/<project>/settings/secrets/actions`:
1. DOCKER_IMAGE_NAME - The name of the Docker image for uploading to the repository.
2. DOCKER_USERNAME - The username for the Docker repository on https://hub.docker.com/.
3. DOCKER_PASSWORD - The password for the Docker repository.
4. AWS_ACCESS_KEY_ID - AWS Access Key ID. https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html
5. AWS_SECRET_ACCESS_KEY - AWS Secret Access Key
6. AWS_REGION - AWS region. https://aws.amazon.com/about-aws/global-infrastructure/regions_az/
7. EKS_CLUSTER_ROLE_ARN - The IAM role's ARN in AWS, providing permissions for managing an Amazon EKS Kubernetes cluster.
8. EKS_CLUSTER_NAME - Amazon EKS Kubernetes cluster name.
9. EKS_CLUSTER_NAMESPACE - Amazon EKS Kubernetes cluster namespace.
10. HELM_REPO_URL - `https://<workspace>.github.io/<project>/`

https://docs.github.com/en/actions

You can run your GitHub Actions locally using https://github.com/nektos/act.  
Usage example:
```bash
act push -j deploy --secret-file my.secrets
```

# Collaboration guidelines
HIRO uses and requires from its partners [GitFlow with Forks](https://hirodevops.notion.site/GitFlow-with-Forks-3b737784e4fc40eaa007f04aed49bb2e?pvs=4)
