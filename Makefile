build:
	docker build --platform linux/amd64 -f docker/Dockerfile.job -t carbon-forecast-job .

run:
	docker run --platform linux/amd64 -v ~/.config/gcloud:/root/.config/gcloud carbon-forecast-job