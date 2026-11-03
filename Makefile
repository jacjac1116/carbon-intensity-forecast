IMAGE = carbon-forecast-job
REMOTE = europe-west2-docker.pkg.dev/gen-lang-client-0232862267/carbon-forecast/carbon-forecast-job:latest


build:
	docker build --platform linux/amd64 -f docker/Dockerfile.job -t $(IMAGE) -t $(REMOTE) .

run:
	docker run --platform linux/amd64 -v ~/.config/gcloud:/root/.config/gcloud $(IMAGE)

push:
	docker push $(REMOTE)

logs:
	gcloud run jobs logs read carbon-forecast-daily --region=europe-west2

fire:
	gcloud scheduler jobs run carbon-forecast-daily-trigger --location=europe-west2


# make build -> rebuild the image from current source
# make run -> optional — test locally first, writes real rows
# make push -> upload to Artifact Registry, replacing :latest
# make fire -> trigger the Cloud Run Job to confirm it works up there