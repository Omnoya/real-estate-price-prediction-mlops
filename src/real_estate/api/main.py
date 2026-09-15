"""FastAPI application factory for frozen CatBoost V1 inference."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from real_estate.api.schemas import (
    HealthResponse,
    ModelResponse,
    PredictionRequest,
    PredictionResponse,
)
from real_estate.api.service import (
    BundleModelUnavailableError,
    ModelServiceError,
    ModelUnavailableError,
    PredictionService,
    load_configured_prediction_service,
)


def create_app(
    service: PredictionService | None = None,
    *,
    service_loader: Callable[[], PredictionService] = load_configured_prediction_service,
) -> FastAPI:
    """Create an app with deferred loading or an injected synthetic service."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.prediction_service = service
        application.state.model_available = service is not None
        if service is None:
            try:
                application.state.prediction_service = service_loader()
                application.state.model_available = True
            except BundleModelUnavailableError:
                raise
            except ModelUnavailableError:
                application.state.prediction_service = None
                application.state.model_available = False
        yield

    application = FastAPI(
        title="Real Estate Price Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def require_service() -> PredictionService:
        current = application.state.prediction_service
        if current is None or not application.state.model_available:
            raise HTTPException(status_code=503, detail="Model service unavailable.")
        return current

    Service = Annotated[PredictionService, Depends(require_service)]

    @application.get("/health", response_model=HealthResponse)
    def health(current: Service) -> HealthResponse:
        return HealthResponse(status="ok", model_version=current.metadata.model_version)

    @application.get("/model", response_model=ModelResponse)
    def model(current: Service) -> ModelResponse:
        return ModelResponse(**vars(current.metadata))

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        request: PredictionRequest,
        current: Service,
    ) -> PredictionResponse:
        try:
            price = current.predict_price_m2(request)
        except ModelServiceError:
            raise HTTPException(status_code=500, detail="Model prediction failed.") from None
        return PredictionResponse(
            predicted_price_m2=price,
            model_version=current.metadata.model_version,
        )

    return application


app = create_app()
