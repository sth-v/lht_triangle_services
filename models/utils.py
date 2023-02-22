
def paginate(data_):
    data = zip(*data_.values())
    while True:
        try:
            yield dict(zip(data_.keys(), next(data)))
        except StopIteration as err:
            break






def decode_sv(data):
    floor = list(data.keys())[0].upper()
    marks = []
    for i in list(data.values())[0]:
        (x, y, z), (typ_name,) = i
        marks.append(dict(x=x, y=y, z=z, tag=typ_name, floor=floor))
    return marks
