import pytest

from backend.storage.disk_counter import DiskCounter


def test_cuenta_lecturas_y_escrituras():
    c = DiskCounter()
    c.record_read(4096)
    c.record_read(4096)
    c.record_write(4096)
    assert (c.disk_reads, c.disk_writes) == (2, 1)
    assert (c.bytes_read, c.bytes_written) == (8192, 4096)


def test_reset_vuelve_a_cero():
    c = DiskCounter()
    c.record_read(4096)
    c.reset()
    assert c.disk_reads == 0 and c.bytes_read == 0


def test_measure_entrega_deltas_no_acumulados():
    c = DiskCounter()
    c.record_read(4096)  # ruido previo
    with c.measure() as m:
        c.record_read(4096)
        c.record_write(4096)
    assert m.metrics.disk_reads == 1
    assert m.metrics.disk_writes == 1
    assert c.disk_reads == 2  # el acumulado sigue intacto


def test_measure_reporta_tiempo_no_negativo():
    c = DiskCounter()
    with c.measure() as m:
        pass
    assert m.metrics.elapsed_ms >= 0.0


def test_measure_publica_metricas_aunque_falle_la_operacion():
    """El endpoint REST debe poder reportar el I/O de una consulta que exploto."""
    c = DiskCounter()
    box = None
    with pytest.raises(RuntimeError):
        with c.measure() as m:
            box = m
            c.record_read(4096)
            raise RuntimeError("consulta invalida")
    assert box.metrics.disk_reads == 1


def test_measure_anidado_mide_rangos_independientes():
    c = DiskCounter()
    with c.measure() as externo:
        c.record_read(4096)
        with c.measure() as interno:
            c.record_read(4096)
            c.record_read(4096)
    assert interno.metrics.disk_reads == 2
    assert externo.metrics.disk_reads == 3
