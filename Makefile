IMAGE = carbon-forecast-job
REGISTRY_IMAGE = europe-west2-docker.pkg.dev/gen-lang-client-0232862267/carbon-forecast/carbon-forecast-job
TAG ?= $(shell git rev-parse HEAD)


build:
	docker build --platform linux/amd64 -f docker/Dockerfile.job -t $(IMAGE) -t $(REMOTE) .

run:
	docker run --platform linux/amd64 -v ~/.config/gcloud:/root/.config/gcloud $(IMAGE)

deploy:
	gcloud artifacts docker images describe $(REGISTRY_IMAGE):$(TAG) > /dev/null
	gcloud run jobs update carbon-forecast-daily \
	  --region europe-west2 \
	  --image $(REGISTRY_IMAGE):$(TAG)

logs:
	gcloud run jobs logs read carbon-forecast-daily --region=europe-west2

fire:
	gcloud scheduler jobs run carbon-forecast-daily-trigger --location=europe-west2


# make build -> rebuild the image from current source
# make run -> optional — test locally first, writes real rows
# make deploy -> deploy the commit I'm on
# make deploy TAG=0206c94 -> roll back to the October image
# make fire -> trigger the Cloud Run Job to confirm it works up there