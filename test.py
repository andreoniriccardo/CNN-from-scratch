def test_generator(numeri):
    for i in range(numeri):
        if i%2 != 0:
            yield "qualcosa", i


test = test_generator(13)
print(type(list(test)[0]))
