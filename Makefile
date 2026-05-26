.PHONY: help backend backend-download backend-asr desktop desktop-install

BACKEND_ARGS ?=
START_BACKEND := ./scripts/start_backend.sh

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make backend            Start full backend with cached local models' \
	  '  make backend-download   Start full backend and allow model downloads' \
	  '  make backend-asr        Start ASR only, without translation or timestamps' \
	  '  make desktop            Start the desktop client' \
	  '  make desktop-install    Install desktop dependencies'

backend:
	$(START_BACKEND) $(BACKEND_ARGS)

backend-download:
	FUNYI_ALLOW_DOWNLOADS=1 $(START_BACKEND) $(BACKEND_ARGS)

backend-asr:
	FUNYI_TRANSLATION_MODEL= FUNYI_TIMESTAMP_MODEL= $(START_BACKEND) $(BACKEND_ARGS)

desktop:
	cd desktop && corepack pnpm run dev

desktop-install:
	cd desktop && corepack pnpm install
