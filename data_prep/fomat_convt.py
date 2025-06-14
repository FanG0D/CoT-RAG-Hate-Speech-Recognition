# 四元组-->三元组
def output2triple(text):
    triple = ''
    seqs = text.split(' [SEP] ')
    for seq in seqs:
        parts = seq.split(' | ')
        triple += f'{parts[0]} | {parts[1]} | {parts[2]} [SEP] '
    return triple[:-7] + ' [END]'

# 三元组-->四元组
def process_triple(output):
    res = ''
    try:
        assert output[-6:] == ' [END]'
        output = output[:-6]
        seqs = output.split(' [SEP] ')
        for seq in seqs:
            parts = seq.split(' | ')
            res += f'{parts[0]} | {parts[1]} | {parts[2]} | '
            hate_classes = parts[2].split(', ')
            hate = 'hate'
            for hate_class in hate_classes:
                if hate_class not in ['Racism', 'Region', 'LGBTQ', 'Sexism', 'others', 'non-hate']:
                    return ''
                if hate_class == 'non-hate':
                    hate = 'non-hate'
            res += f'{hate} [SEP] '
        res = res[:-7] + ' [END]'
        return res
    except:
        return ''

# 判断四元组是否为符合要求
def check_response(output):
    try:
        assert output[-6:] == ' [END]'
        output = output[:-6]
        seqs = output.split(' [SEP] ')
        for seq in seqs:
            parts = seq.split(' | ')
            hate_classes = parts[2].split(', ')
            hate = parts[3]
            for hate_class in hate_classes:
                if hate_class not in ['Racism', 'Region', 'LGBTQ', 'Sexism', 'others', 'non-hate']:
                    return False
                if hate_class == 'non-hate':
                    if hate != 'non-hate':
                        return False
                else:
                    if hate != 'hate':
                        return False
        return True
    except:
        return False