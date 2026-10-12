# syntax=docker/dockerfile:1.10
#
# glasswing-server with its models baked in. Every input is pinned: base images by digest, model
# files by sha256 (ADD --checksum fails the build if a download differs by a single byte).

# --- build -------------------------------------------------------------------------------------
FROM rust:1.92-slim-trixie@sha256:bf3368a992915f128293ac76917ab6e561e4dda883273c8f5c9f6f8ea37a378e AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p glasswing-server

# --- models ------------------------------------------------------------------------------------
# SmolLM2-135M-Instruct (Apache-2.0), revision 12fd25f77366fa6b3b4b768ec3050bf629380bac.
FROM scratch AS models
ARG HF=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac
ADD --checksum=sha256:8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a $HF/config.json /models/smollm2/config.json
ADD --checksum=sha256:9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c $HF/tokenizer.json /models/smollm2/tokenizer.json
ADD --checksum=sha256:5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c $HF/model.safetensors /models/smollm2/model.safetensors
# glasswing, from this repo's release (the same files the browser downloads).
ARG GW=https://github.com/tmukh/glasswing/releases/download/v0.1.0
ADD --checksum=sha256:GLASSWING_WEIGHTS_SHA256 $GW/model-f16.safetensors /models/glasswing/model-f16.safetensors
COPY assets/tokenizer.json /models/glasswing/tokenizer.json

# --- run ---------------------------------------------------------------------------------------
# Distroless: glibc and the binary, no shell, no package manager. Numeric non-root user, so
# Kubernetes' runAsNonRoot can verify it.
FROM gcr.io/distroless/cc-debian13:nonroot@sha256:e792ab3d241a468a4fd7519ddbbebe66b49b5f365771716ea688ad40b6c6f1c2
COPY --from=build /src/target/release/glasswing-server /usr/local/bin/glasswing-server
COPY --from=models /models /models
COPY LICENSE /usr/share/doc/glasswing/LICENSE
ENV PORT=8080 \
    GLASSWING_SMOLLM2_DIR=/models/smollm2 \
    GLASSWING_WEIGHTS=/models/glasswing/model-f16.safetensors \
    GLASSWING_TOKENIZER=/models/glasswing/tokenizer.json
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/glasswing-server"]
