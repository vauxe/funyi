.PHONY: help backend backend-download backend-asr desktop desktop-install desktop-lint desktop-format desktop-format-check desktop-check

BACKEND_ARGS ?=
START_BACKEND := ./scripts/start_backend.sh
DESKTOP_PNPM := cd desktop && corepack pnpm

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make backend            Start full backend with cached local models' \
	  '  make backend-download   Start full backend and allow model downloads' \
	  '  make backend-asr        Start ASR plus aligner, without translation' \
	  '  make desktop            Start the desktop client' \
	  '  make desktop-install    Install desktop dependencies' \
	  '  make desktop-lint       Run desktop lint gates' \
	  '  make desktop-format     Format desktop code' \
	  '  make desktop-check      Run desktop lint, format, type, and test gates'

backend:
	$(START_BACKEND) $(BACKEND_ARGS)

backend-download:
	FUNYI_ALLOW_DOWNLOADS=1 $(START_BACKEND) $(BACKEND_ARGS)

backend-asr:
	FUNYI_TRANSLATION_MODEL= $(START_BACKEND) $(BACKEND_ARGS)

desktop:
	$(DESKTOP_PNPM) run dev

desktop-install:
	$(DESKTOP_PNPM) install

desktop-lint:
	$(DESKTOP_PNPM) run lint

desktop-format:
	$(DESKTOP_PNPM) run format

desktop-format-check:
	$(DESKTOP_PNPM) run format:check

desktop-check:
	$(DESKTOP_PNPM) run check
