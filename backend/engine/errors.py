"""Jerarquia de errores del motor.

Separar SQLSyntaxError de SQLRuntimeError importa para la API: el primero es un
400 (el cliente escribio mal la consulta), el segundo puede ser un 400 o un 409
segun el caso. NotImplementedFeature es un 501 y existe para que el frontend
distinguja "esto esta mal" de "esto todavia no esta construido".
"""


class EngineError(Exception):
    """Base de todos los errores del motor."""


class SQLSyntaxError(EngineError):
    """La consulta no se pudo parsear."""


class SQLRuntimeError(EngineError):
    """La consulta es valida pero fallo al ejecutarse."""


class TableNotFound(SQLRuntimeError):
    pass


class TableAlreadyExists(SQLRuntimeError):
    pass


class ColumnNotFound(SQLRuntimeError):
    pass


class NotImplementedFeature(EngineError):
    """Funcionalidad reconocida por el parser pero aun no construida.

    Se usa para Heap File, B+ Tree y Hash Dinamico mientras los implementan los
    demas integrantes del equipo. El mensaje debe decir QUE falta, no solo que
    fallo, porque llega tal cual a la interfaz.
    """
