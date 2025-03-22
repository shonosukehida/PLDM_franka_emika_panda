def hex_to_rgba(hex_color, alpha=1.0):
    # '#' を除去
    hex_color = hex_color.lstrip('#')
    
    # RGBをそれぞれ10進数に変換し、0〜1に正規化
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    return (round(r, 3), round(g, 3), round(b, 3), alpha)


def main():
    #色指定
    hex_color = '#4d291a'


    print(hex_to_rgba(hex_color))
    return 

if __name__=='__main__':
    main()