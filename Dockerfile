FROM golang:1.25-alpine AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /gpucalc .

FROM gcr.io/distroless/static-debian12
COPY --from=build /gpucalc /gpucalc
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["/gpucalc", "-addr", ":8080", "-config", "/data/config.json"]
