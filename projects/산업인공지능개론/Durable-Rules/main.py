from durable.lang import *

with ruleset('hydrogen_system'):

    # 압축기
    @when_all((m.facility == 'compressor') & (m.sensor == 'pressure') & (m.value < 60))
    def compressor_low_pressure(c):
        c.assert_fact({'type': 'maintenance', 'message': '압축기에서 저압 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'compressor') & (m.sensor == 'pressure') & (m.value > 120))
    def compressor_high_pressure(c):
        c.assert_fact({'type': 'maintenance', 'message': '압축기에서 고압 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'compressor') & (m.sensor == 'temperature') & (m.value < -30))
    def compressor_low_temperature(c):
        c.assert_fact({'type': 'maintenance', 'message': '압축기에서 저온 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'compressor') & (m.sensor == 'temperature') & (m.value > 10))
    def compressor_high_temperature(c):
        c.assert_fact({'type': 'maintenance', 'message': '압축기에서 고온 데이터가 감지되었습니다.'})

    # 저장용기
    @when_all((m.facility == 'storage') & (m.sensor == 'pressure') & (m.value < 20))
    def storage_low_pressure(c):
        c.assert_fact({'type': 'maintenance', 'message': '저장용기에서 저압 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'storage') & (m.sensor == 'pressure') & (m.value > 80))
    def storage_high_pressure(c):
        c.assert_fact({'type': 'maintenance', 'message': '저장용기에서 고압 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'storage') & (m.sensor == 'temperature') & (m.value < -30))
    def storage_low_temperature(c):
        c.assert_fact({'type': 'maintenance', 'message': '저장용기에서 저온 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'storage') & (m.sensor == 'temperature') & (m.value > 10))
    def storage_high_temperature(c):
        c.assert_fact({'type': 'maintenance', 'message': '저장용기에서 고온 데이터가 감지되었습니다.'})

    # 충전기
    @when_all((m.facility == 'dispenser') & (m.sensor == 'pressure') & (m.value < 30))
    def storage_low_pressure(c):
        c.assert_fact({'type': 'maintenance', 'message': '충전기에서 저압 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'dispenser') & (m.sensor == 'pressure') & (m.value > 100))
    def storage_high_pressure(c):
        c.assert_fact({'type': 'maintenance', 'message': '충전기에서 고압 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'dispenser') & (m.sensor == 'temperature') & (m.value < -30))
    def storage_low_temperature(c):
        c.assert_fact({'type': 'maintenance', 'message': '충전기에서 저온 데이터가 감지되었습니다.'})

    @when_all((m.facility == 'dispenser') & (m.sensor == 'temperature') & (m.value > 10))
    def storage_high_temperature(c):
        c.assert_fact({'type': 'maintenance', 'message': '충전기에서 고온 데이터가 감지되었습니다.'})

    # 출력문
    @when_all(+m.facility)
    def log_output(c):
        print('{0} > {1}, Value: {2}'.format(c.m.facility, c.m.sensor, c.m.value))

    # 이상 범위일 경우 출력
    @when_all(+m.type)
    def log_maintenance(c):
        print('Maintenance: {0}'.format(c.m.message))

# 압축기 데이터
assert_fact('hydrogen_system', {'facility': 'compressor', 'sensor': 'pressure', 'value': 10})
assert_fact('hydrogen_system', {'facility': 'compressor', 'sensor': 'pressure', 'value': 70})
assert_fact('hydrogen_system', {'facility': 'compressor', 'sensor': 'pressure', 'value': 140})
assert_fact('hydrogen_system', {'facility': 'compressor', 'sensor': 'temperature', 'value': -50})
assert_fact('hydrogen_system', {'facility': 'compressor', 'sensor': 'temperature', 'value': 0})
assert_fact('hydrogen_system', {'facility': 'compressor', 'sensor': 'temperature', 'value': 20})

# 저장용기 데이터
assert_fact('hydrogen_system', {'facility': 'storage', 'sensor': 'pressure', 'value': 15})
assert_fact('hydrogen_system', {'facility': 'storage', 'sensor': 'pressure', 'value': 60})
assert_fact('hydrogen_system', {'facility': 'storage', 'sensor': 'pressure', 'value': 200})
assert_fact('hydrogen_system', {'facility': 'storage', 'sensor': 'temperature', 'value': -50})
assert_fact('hydrogen_system', {'facility': 'storage', 'sensor': 'temperature', 'value': 0})
assert_fact('hydrogen_system', {'facility': 'storage', 'sensor': 'temperature', 'value': 20})

# 충전기 데이터
assert_fact('hydrogen_system', {'facility': 'dispenser', 'sensor': 'pressure', 'value': 5})
assert_fact('hydrogen_system', {'facility': 'dispenser', 'sensor': 'pressure', 'value': 20})
assert_fact('hydrogen_system', {'facility': 'dispenser', 'sensor': 'pressure', 'value': 110})
assert_fact('hydrogen_system', {'facility': 'dispenser', 'sensor': 'temperature', 'value': -50})
assert_fact('hydrogen_system', {'facility': 'dispenser', 'sensor': 'temperature', 'value': 0})
assert_fact('hydrogen_system', {'facility': 'dispenser', 'sensor': 'temperature', 'value': 20})
