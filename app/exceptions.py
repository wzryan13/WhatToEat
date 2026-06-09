from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import ErrorResponse


class AppException(Exception):
    """业务异常基类。endpoint 只 raise，统一由全局 handler 转成响应。"""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "服务器内部错误"

    def __init__(self, message: str | None = None):
        if message is not None:
            self.message = message
        super().__init__(self.message)


class NotFoundError(AppException):
    status_code = 404
    code = "not_found"
    message = "资源不存在"


class ConflictError(AppException):
    status_code = 409
    code = "conflict"
    message = "资源冲突"


class ServiceUnavailableError(AppException):
    status_code = 503
    code = "service_unavailable"
    message = "服务暂不可用"


def register_exception_handlers(app: FastAPI) -> None:
    # raise NotFoundError("该菜品不存在") 后中断，fastapi接管，识别为AppException子类统一返回格式
    @app.exception_handler(AppException)
    async def _handle_app_exception(_: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(code=exc.code, message=exc.message).model_dump(),
        )
