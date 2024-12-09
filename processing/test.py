def convert_to_multi_base(number, bases):
    # Calculate max possible value
    max_value = 1
    for base in bases:
        max_value *= base
    max_value -= 1
    
    # Validate number range
    if number < 0 or number > max_value:
        raise ValueError("Number out of representable range")
    
    # Initialize result array with same length as bases
    result = [0] * len(bases)
    
    # Convert number, working from rightmost base
    for i in range(len(bases) - 1, -1, -1):
        base = bases[i]
        result[i] = number % base
        number //= base
    
    return result

bases = [2, 10, 3]
print(f"Max value: {2*10*3 - 1}")  # Should be 59
for i in range(61):
    print(f"{i}: {convert_to_multi_base(i, bases)}")