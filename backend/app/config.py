from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://synthflow:synthflow@localhost:5432/synthflow"

    # Redis
    redis_url: str = "redis://default:@localhost:6379"

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # Security
    encryption_key: str = "change-me-in-production"

    # CORS — restrict to frontend origin in production
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Payment providers (optional — graceful fallback when empty)
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # App
    app_name: str = "SynthFlow"
    app_version: str = "2.0.0"
    debug: bool = False


settings = Settings()
