# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
#from PIL import Image
#import numpy


import math, operator
from PIL import Image, ImageChops
from PIL import ImageOps, ImageStat, ImageFilter


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.

    object_size = {"very small": 0, "small": 1, "medium": 2, "large": 3, "very large": 4, "huge": 5}
    object_width = {"very small": 0, "small": 1, "medium": 2, "large": 3, "very large": 4, "huge": 5}
    object_height = {"very small": 0, "small": 1, "medium": 2, "large": 3, "very large": 4, "huge": 5}
    position = ['left-of', 'right-of', 'inside', 'above', 'overlaps']
    def Solve(self,problem):
        # For project1, I use verbal representation to solve the problems
        # I use semantic networks as the representation of the problem, each object was identified based on their attributes.
        # The transformation will be stored the in the dictionary called difference.
        # Next, the agent will iterate through the solutions 1-6, put into position D, compare the transformation from C to D with A to B (or compare B to D with A to C) and find the best match.

        print problem.name + ', ' + problem.problemType
        difference = {}

        if problem.problemType == '2x2':
            # read figure A,B,C,D compare AB and AC, find the differences.
            A = problem.figures["A"]
            B = problem.figures["B"]
            C = problem.figures["C"]
            difference['AB'] = self.get_diff(A, B)
            difference['AC'] = self.get_diff(A, C)
            # for each answer (1-6), check if the differences between B/C and current answer are similar compared to the differences between AB or AC
            answers = []
            for i in range(1, 7):
                curr = problem.figures[str(i)]
                difference['Bcurr'] = self.get_diff(B, curr)
                difference['Ccurr'] = self.get_diff(C, curr)
                # If the differences are the same in each row and each column, the answer is perfect. return the answer
                if difference['AB'] == difference['Ccurr'] and difference['AC'] == difference['Bcurr']:
                    return i
                # else there might be multiple answers
                elif difference['AB'] == difference['Ccurr'] or difference['AC'] == difference['Bcurr']:
                    answers.append(i)
            if len(answers) < 1:
                # if we cannot find answer by using the verbal transformation, we will look for the answer using visual transformation
                # get visual score and compare the minimum differences

                diff_visual = {}
                best_score = 10000000
                answer = -1
                difference = {}
                difference['AB'] = self.visual_score(A, B)
                difference['AC'] = self.visual_score(A, C)
                for i in range(1, 7):
                    curr = problem.figures[str(i)]
                    difference['Bcurr'] = self.visual_score(B, curr)
                    difference['Ccurr'] = self.visual_score(C, curr)
                    diff_visual[i] = abs(difference['AB'] - difference['Ccurr']) + abs(
                        difference['AC'] - difference['Bcurr'])
                    if diff_visual[i] < best_score:
                        answer = i
                        best_score = min(diff_visual[i], best_score)

                if answer != -1:
                    print "According to the visual score, we got answer: "
                    return answer
                else:
                    print "Failed to find answer"
                    return -1
            print "result is: ", answers[0]
            return answers[0]

        if problem.problemType == '3x3':

            A = problem.figures["A"]
            B = problem.figures["B"]
            C = problem.figures["C"]
            D = problem.figures["D"]
            E = problem.figures["E"]
            F = problem.figures["F"]
            G = problem.figures["G"]
            H = problem.figures["H"]
            img_a = Image.open((problem.figures["A"]).visualFilename).convert('1')
            img_b = Image.open((problem.figures["B"]).visualFilename).convert('1')
            img_c = Image.open((problem.figures["C"]).visualFilename).convert('1')
            img_d = Image.open((problem.figures["D"]).visualFilename).convert('1')
            img_e = Image.open((problem.figures["E"]).visualFilename).convert('1')
            img_f = Image.open((problem.figures["F"]).visualFilename).convert('1')
            img_g = Image.open((problem.figures["G"]).visualFilename).convert('1')
            img_h = Image.open((problem.figures["H"]).visualFilename).convert('1')
            difference['AB'] = self.get_diff(A, B)
            difference['AE'] = self.get_diff(A, E)
            difference['BC'] = self.get_diff(B, C)
            difference['DE'] = self.get_diff(D, E)
            difference['EF'] = self.get_diff(E, F)
            difference['DF'] = self.get_diff(D, F)
            difference['GH'] = self.get_diff(G, H)
            difference['AD'] = self.get_diff(B, D)
            difference['DG'] = self.get_diff(D, G)
            difference['BE'] = self.get_diff(B, E)
            difference['EH'] = self.get_diff(E, H)
            difference['CF'] = self.get_diff(C, F)
            answers = []
            darkpixelA = self.dark_pixel(A)
            darkpixelB = self.dark_pixel(B)
            darkpixelC = self.dark_pixel(C)
            darkpixelD = self.dark_pixel(D)
            darkpixelE = self.dark_pixel(E)
            darkpixelF = self.dark_pixel(F)
            darkpixelG = self.dark_pixel(G)
            darkpixelH = self.dark_pixel(H)

            darkpixelupperA = self.dark_pixel_upperhalf(A)
            darkpixelupperB = self.dark_pixel_upperhalf(B)
            darkpixelupperC = self.dark_pixel_upperhalf(C)
            darkpixelowerA = self.dark_pixel_lowerhalf(A)
            darkpixelowerB = self.dark_pixel_lowerhalf(B)
            darkpixelupperG = self.dark_pixel_upperhalf(G)
            darkpixelowerH = self.dark_pixel_lowerhalf(H)
            darkpixelupperH = self.dark_pixel_upperhalf(H)
            darkpixelupperD = self.dark_pixel_upperhalf(D)
            darkpixelupperE = self.dark_pixel_upperhalf(E)
            darkpixelowerD = self.dark_pixel_lowerhalf(D)
            darkpixelowerE = self.dark_pixel_lowerhalf(E)
            darkpixelowerG = self.dark_pixel_lowerhalf(G)

            darkpixeleftG = self.dark_pixel_lefthalf(G)

            a_xor_b = ImageChops.logical_xor(img_a, img_b)
            d_xor_e = ImageChops.logical_xor(img_d, img_e)
            g_xor_h = ImageChops.logical_xor(img_g, img_h)

            a_and_b = ImageChops.logical_and(img_a, img_b)
            a_and_c = ImageChops.logical_and(img_a, img_c)
            d_and_e = ImageChops.logical_and(img_d, img_e)
            d_and_f = ImageChops.logical_and(img_d, img_f)
            g_and_h = ImageChops.logical_and(img_g, img_h)
            g_and_e = ImageChops.logical_and(img_g, img_e)
            a_and_f = ImageChops.logical_and(img_a, img_f)

            a_or_b = ImageChops.logical_or(img_a, img_b)
            d_or_e = ImageChops.logical_or(img_d, img_e)
            g_or_h = ImageChops.logical_or(img_g, img_e)

            figures_darkpixel = [darkpixelA, darkpixelB, darkpixelC, darkpixelD, darkpixelE, darkpixelF, darkpixelG,
                                 darkpixelH]
            figures_darkpixel_set = set(figures_darkpixel)
            count1, count2, count3 = 0, 0, 0
            pixel1, pixel2, pixel3 = 0, 0, 0
            if (len(figures_darkpixel_set) > 2):
                newlist = list(figures_darkpixel_set)
                pixel1, pixel2, pixel3 = newlist[0], newlist[1], newlist[2]

            for i in range(len(figures_darkpixel)):
                if figures_darkpixel[i] == pixel1:
                    count1 += 1
                elif figures_darkpixel[i] == pixel2:
                    count2 += 1
                elif figures_darkpixel[i] == pixel3:
                    count3 += 1


            if problem.hasVerbal:
                for i in range(1, 9):
                    curr = problem.figures[str(i)]
                    difference['Fcurr'] = self.get_diff(F, curr)
                    difference['Hcurr'] = self.get_diff(H, curr)
                    difference['Ecurr'] = self.get_diff(E, curr)
                    # Check horizontal and vertical transformation
                    if difference['CF'] == difference['Fcurr'] or difference['GH'] == difference['Hcurr'] or difference[
                        'AE'] == difference['Ecurr'] or difference['BC'] == difference['Hcurr']:
                        answers.append(i)

                if len(answers) == 1:

                    print "result is: ", answers[0]
                    return answers[0]

                elif len(answers) > 1:
                    best_answer=-2
                    for i in answers:
                        curr = problem.figures[str(i)]
                        difference['Fcurr'] = self.get_diff(F, curr)
                        difference['Hcurr'] = self.get_diff(H, curr)
                        difference['Ecurr'] = self.get_diff(E, curr)
                        if difference['CF'] == difference['Fcurr'] and difference['GH'] == difference['Hcurr'] and difference[
                        'AE'] != difference['Ecurr']:
                            best_answer = i
                            return best_answer

                    if best_answer == -2:
                        for i in answers:
                            curr = problem.figures[str(i)]

                            difference['Hcurr'] = self.get_diff(H, curr)

                            if difference['BC'] ==  difference['Hcurr']:
                                best_answer = i
                                return best_answer

                    best_answer = -1


                    if best_answer == -1:

                        if darkpixelowerG!=0 and darkpixelupperA/darkpixelowerG > 0.95 and darkpixelupperA/darkpixelowerG < 1.05 and darkpixelowerH!=0 and darkpixelupperB/darkpixelowerH > 0.95 and darkpixelupperB/darkpixelowerH < 1.05:
                            for i in range(1, 9):
                                I = problem.figures[str(i)]
                                darkpixelowerI = self.dark_pixel_lowerhalf(I)
                                darkpixelrightI = self.dark_pixel_righthalf(I)
                                if darkpixelowerI!=0 and darkpixelupperC/darkpixelowerI > 0.95 and darkpixelupperC/darkpixelowerI < 1.05 and darkpixelrightI!=0 and darkpixeleftG/darkpixelrightI > 0.95 and darkpixeleftG/darkpixelrightI < 1.05:
                                    return i


                        visualCF = self.find_dark_ratio(C, F)

                        index = -1
                        best_score = 1
                        for i in range(1, 9):
                            curr = problem.figures[str(i)]
                           
                            visualFcurr = self.find_dark_ratio(F, curr)
                            print "current_score" + str(visualFcurr)
                            if abs(visualFcurr - visualCF) < best_score:
                                best_score = abs(visualFcurr - visualCF)
                                index = i
                        if index != -1:
                            return index
                        else:
                            return -1

                # GET visual score
                else:
                    print "answers I got using verbal representation are"
                    print answers
                    print " now using visual representation to find better solution"
                    visualBC = self.find_dark_ratio(B, C)
                    visualGH = self.find_dark_ratio(G, H)

                    visualDG = self.find_dark_ratio(D, G)
                    visualCF = self.find_dark_ratio(C, F)

                    index = -1
                    best_score = 1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        visualHcurr = self.find_dark_ratio(H, curr)
                        visualFcurr = self.find_dark_ratio(F, curr)
                        print "current_score" + str(visualFcurr)
                        if abs(visualFcurr - visualCF) < best_score:
                            best_score = abs(visualFcurr - visualCF)
                            index = i
                    if index != -1:
                        return index
                    else:
                        return -1

            else:
                # find transformation of basic D and E
                transformation = ""
                sum_dark_pixelC = darkpixelA + darkpixelB
                minus_dark_pixelC = darkpixelA - darkpixelB
                minus_dark_pixelF = darkpixelD - darkpixelE

                if darkpixelA == darkpixelB == darkpixelC:
                    transformation = "same"

                candidate_pixel = -1
                if count1 + count2 + count3 == 8:

                    if count1 == 2:
                        candidate_pixel = pixel1
                    elif count2 == 2:
                        candidate_pixel = pixel2
                    else:
                        candidate_pixel = pixel3
                    transformation = "reuse"
                    # D11 diagonal image is same
                if transformation != "same" and transformation != "reuse":
                    if float(darkpixelA) / float(darkpixelE) < 1.01 and float(darkpixelA) / float(darkpixelE) > 0.99:
                        transformation = "diagonal same"

                dark_pixel_difference = 1000
                dark_pixel_index = -1

                if float(darkpixelC) / float(sum_dark_pixelC) > 0.96 and float(darkpixelC) / float(
                        sum_dark_pixelC) < 1.05:
                    transformation = "c=a+b"

                if self.isxor(a_xor_b, img_c) > 0.9 and self.isxor(a_xor_b, img_c) < 1.1 and self.isxor(d_xor_e, img_f) > 0.9 and self.isxor(d_xor_e, img_f) < 1.1:
                    transformation = "xor"

                if self.isand(a_and_b, img_c) > 0.9 and self.isand(a_and_b, img_c) < 1.1 and self.isand(d_and_e,
                                                                                                          img_f) > 0.9 and self.isand(
                        d_and_e, img_f) < 1.1:
                    transformation = "and"

                if self.isor(a_or_b, img_c) > 0.9 and self.isor(a_or_b, img_c) < 1.1 and self.isor(d_or_e,
                                                                                                     img_f) > 0.9 and self.isor(
                        d_or_e, img_f) < 1.1:
                    transformation = "or"


                if transformation != "reuse" and transformation != "same" and transformation != "diagonal same" and transformation != "c=a+b" and transformation != "xor" and transformation != "and" and transformation != "or":
                    component1, component2, component3, component4 = 0, 0, 0, 0
                    print "minus"
                    print self.dark_pixel_lowerhalf(D) - self.dark_pixel_upperhalf(E)
                    print "minus result"
                    print self.dark_pixel_lowerhalf(F)
                    if self.find_dark_ratio(D, G) > 1:
                        component1 = self.find_dark_ratio(D, G)
                    else:
                        component1 = 1 / float(self.find_dark_ratio(D, G))

                    if self.find_dark_ratio(G, H) > 1:
                        component2 = self.find_dark_ratio(G, H)
                    else:
                        component2 = 1 / float(self.find_dark_ratio(G, H))

                    if self.find_dark_ratio(A, E) > 1:
                        component3 = self.find_dark_ratio(A, E)
                    else:
                        component3 = 1 / float(self.find_dark_ratio(A, E))

                    if self.find_dark_ratio(B, D) > 1:
                        component4 = self.find_dark_ratio(B, D)
                    else:
                        component4 = 1 / float(self.find_dark_ratio(B, D))

                    pixle_difference_list = [component1, component2, component3, component4]
                    min_pixel_difference = min(pixle_difference_list)

                    if float(darkpixelC) / float(darkpixelupperA + darkpixelowerB) > 0.95 and float(
                            darkpixelC) / float(darkpixelupperA + darkpixelowerB) < 1.05 and float(darkpixelF) / float(
                                darkpixelupperD + darkpixelowerE) > 0.95 and float(
                        darkpixelF) / float(darkpixelupperD + darkpixelowerE) < 1.05:
                        transformation = "half-half"

                    elif self.dark_pixel_lowerhalf(C)!=0 and self.dark_pixel_lowerhalf(F)!=0 and abs(self.dark_pixel_lowerhalf(A) - self.dark_pixel_upperhalf(B))/float(self.dark_pixel_lowerhalf(C)) > 0.85 and abs(self.dark_pixel_lowerhalf(A) - self.dark_pixel_upperhalf(B))/float(self.dark_pixel_lowerhalf(C))< 1.15 and abs(self.dark_pixel_lowerhalf(D) - self.dark_pixel_upperhalf(E))/float(self.dark_pixel_lowerhalf(F))< 1.17 and abs(self.dark_pixel_lowerhalf(D) - self.dark_pixel_upperhalf(E))/float(self.dark_pixel_lowerhalf(F))>0.83:
                        transformation = "half minus"


                    elif float(darkpixelC) / float(sum_dark_pixelC) > 0.94 and float(darkpixelC) / float(
                            sum_dark_pixelC) < 1.07:
                        transformation = "c=a+b"

                    elif float(darkpixelC) / float(minus_dark_pixelC) > 0.93 and float(darkpixelC) / float(
                            minus_dark_pixelC) < 1.07 and float(darkpixelF) / float(minus_dark_pixelF) > 0.9 and float(darkpixelF) / float(
                            minus_dark_pixelF) < 1.1:
                        transformation = "c=a-b"

                    elif self.isand(g_and_e, img_c) > 0.95 and self.isand(g_and_e, img_c) < 1.05 and self.isand(a_and_f,
                                                                                                              img_h) > 0.95 and self.isand(
                        a_and_f, img_h) < 1.05:
                        transformation = "special and"

                    elif min_pixel_difference == pixle_difference_list[0]:
                        transformation = "vertical"


                    elif min_pixel_difference == pixle_difference_list[1]:
                        transformation = "horizontal"
                    elif min_pixel_difference == pixle_difference_list[2]:
                        transformation = "diagnol"
                    elif self.isand(a_and_c, img_b) > 0.9 and self.isand(a_and_c, img_b) < 1.1 and self.isand(d_and_f, img_e) > 0.9 and self.isand(d_and_f, img_e)<1.1:
                        transformation = "center_and"
                    elif min_pixel_difference == pixle_difference_list[3]:
                        transformation = "center"



                    else:
                        transformation = "unsure"
                print "transformation is " + transformation
                print figures_darkpixel
                print "a xor b ratio is " + str(self.isxor(a_xor_b, img_c))
                print "a and b ratio is " + str(self.isand(a_and_b, img_c))
                print "g and e ratio is " + str(self.isand(g_and_e, img_c))
                print "a and f ratio is " + str(self.isand(a_and_f, img_h))
                print "a or b ratio is " + str(self.isor(a_or_b, img_c))

                if transformation == "same":

                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        darkpixelI = self.dark_pixel(curr)
                        if darkpixelI == darkpixelG:
                            return i
                elif transformation == "reuse":
                    print figures_darkpixel
                    print "candidate pixel is " + str(candidate_pixel)
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        darkpixelI = self.dark_pixel(curr)
                        if abs(darkpixelI - candidate_pixel) < dark_pixel_difference:
                            dark_pixel_difference = abs(darkpixelI - candidate_pixel)
                            dark_pixel_index = i
                    return dark_pixel_index

                elif transformation == "vertical":
                    mindiff = 1000
                    vertical_index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        dark_ratio_CF = self.find_dark_ratio(C, F)
                        dark_ratio_Fcurr = self.find_dark_ratio(F, curr)
                        if abs(dark_ratio_Fcurr - dark_ratio_CF) < mindiff:
                            mindiff = abs(dark_ratio_Fcurr - dark_ratio_CF)
                            vertical_index = i
                    return vertical_index

                elif transformation == "diagnol":
                    mindiff = 1000
                    list_images = figures_darkpixel
                    diag_index = -1
                    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
                    print list_images
                    for i in range(1, 9):
                        k = i
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)

                        for x in list_images:
                            if curr_dark_pixel in list_images:
                                if k in candidates:
                                    candidates.remove(k)

                    print candidates

                    for i in candidates:
                        curr = problem.figures[str(i)]
                        dark_ratio_AE = self.find_dark_ratio(A, E)
                        dark_ratio_Ecurr = self.find_dark_ratio(E, curr)
                        print "curr " + str(self.dark_pixel(curr))
                        print "abs store" + str(abs(float(dark_ratio_AE) / float(dark_ratio_Ecurr) - 1))
                        if abs(float(dark_ratio_AE) / float(dark_ratio_Ecurr) - 1) < mindiff:
                            mindiff = abs(float(dark_ratio_AE) / float(dark_ratio_Ecurr) - 1)
                            diag_index = i
                    return diag_index

                elif transformation == "diagonal same":
                    index = -1
                    for i in range(1, 9):

                        curr = problem.figures[str(i)]
                        darkpixelI = self.dark_pixel(curr)
                        print figures_darkpixel
                        print darkpixelI
                        if darkpixelI == darkpixelA or darkpixelI == darkpixelE:
                            index = i
                            return i
                    if index == -1:
                        candidates = []
                        mindiff = 1000
                        for i in range(1, 9):
                            curr = problem.figures[str(i)]
                            curr_dark_pixel = self.dark_pixel(curr)
                            candidates.append(i)
                            for each in figures_darkpixel:
                                if curr_dark_pixel / float(each) > 0.99 and curr_dark_pixel / float(each) < 1.01:
                                    if i in candidates:
                                        candidates.remove(i)
                        print candidates
                        for c in candidates:
                            curr = problem.figures[str(c)]
                            curr_dark_pixel = self.dark_pixel(curr)
                            if darkpixelG - darkpixelD > 0:
                                if curr_dark_pixel - darkpixelF < 0:
                                    continue
                            if darkpixelG - darkpixelD < 0:

                                if curr_dark_pixel - darkpixelF > 0:
                                    continue
                            curr_diff = abs(curr_dark_pixel - darkpixelF) - (darkpixelG - darkpixelD)
                            if curr_diff < mindiff:
                                mindiff = curr_diff
                                index = c

                    return index

                elif transformation == "center":
                    list_images = figures_darkpixel
                    print list_images
                    mindiff = 1000
                    center_index = -1
                    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
                    for i in range(1, 9):
                        k = i
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)

                        for x in list_images:
                            if curr_dark_pixel in list_images:
                                if k in candidates:
                                    candidates.remove(k)

                            if curr_dark_pixel!=0 and float(x) / float(curr_dark_pixel) > 0.98 and float(x) / float(curr_dark_pixel) < 1.05:
                                if k in candidates:
                                    candidates.remove(k)

                    print candidates
                    for j in candidates:
                        curr = problem.figures[str(j)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        dark_ratio_DB = self.find_dark_ratio(D, B)
                        dark_ratio_Dcurr = self.find_dark_ratio(D, curr)
                        print list_images
                        print "curr center pixel " + str(curr_dark_pixel)
                        print "abs store" + str(abs(float(dark_ratio_DB) / float(dark_ratio_Dcurr) - 1))
                        if abs(float(dark_ratio_DB) / float(dark_ratio_Dcurr) - 1) < mindiff:
                            mindiff = abs(float(dark_ratio_DB) / float(dark_ratio_Dcurr) - 1)
                            center_index = j

                    if center_index != -1:
                        return center_index

                    # if center_index == -1 , we need to modify the threshhold to get more candidates
                    if center_index == -1:
                        candidates = [1, 2, 3, 4, 5, 6, 7, 8]
                        for i in range(1, 9):
                            k = i
                            curr = problem.figures[str(i)]
                            curr_dark_pixel = self.dark_pixel(curr)

                            for x in list_images:
                                if curr_dark_pixel in list_images:
                                    if k in candidates:
                                        candidates.remove(k)

                                if float(x) / float(curr_dark_pixel) > 0.99 and float(x) / float(
                                        curr_dark_pixel) < 1.01:
                                    if k in candidates:
                                        candidates.remove(k)

                        print candidates
                        for j in candidates:
                            curr = problem.figures[str(j)]
                            curr_dark_pixel = self.dark_pixel(curr)
                            dark_ratio_DB = self.find_dark_ratio(D, B)
                            dark_ratio_Dcurr = self.find_dark_ratio(D, curr)
                            print list_images
                            print "curr center pixel " + str(curr_dark_pixel)
                            print "abs store" + str(abs(float(dark_ratio_DB) / float(dark_ratio_Dcurr) - 1))
                            if abs(float(dark_ratio_DB) / float(dark_ratio_Dcurr) - 1) < mindiff:
                                mindiff = abs(float(dark_ratio_DB) / float(dark_ratio_Dcurr) - 1)
                                center_index = j

                    return center_index

                    # if transformation is horizontal
                elif transformation == "horizontal":
                    list_images = figures_darkpixel
                    mindiff = 1000
                    horizontal_index = -1
                    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
                    for i in range(1, 9):
                        k = i
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)

                        for x in list_images:
                            if curr_dark_pixel in list_images:
                                if k in candidates:
                                    candidates.remove(k)

                            if float(x) / float(curr_dark_pixel) > 0.98 and float(x) / float(curr_dark_pixel) < 1.05:
                                if k in candidates:
                                    candidates.remove(k)

                    print candidates
                    for j in candidates:
                        curr = problem.figures[str(j)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        dark_ratio_HG = self.find_dark_ratio(H, G)
                        dark_ratio_currH = self.find_dark_ratio(curr, H)
                        print list_images
                        print "curr center pixel " + str(curr_dark_pixel)
                        print "abs store" + str(abs(float(dark_ratio_HG) / float(dark_ratio_currH) - 1))
                        if (abs(float(dark_ratio_HG) / float(dark_ratio_currH) - 1)) < mindiff:
                            mindiff = (abs(float(dark_ratio_HG) / float(dark_ratio_currH) - 1))
                            horizontal_index = j
                    return horizontal_index

                elif transformation == "c=a+b":
                    mindiff = 1000
                    predicted_GH = darkpixelG + darkpixelH
                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        if abs(curr_dark_pixel - predicted_GH) < mindiff:
                            mindiff = abs(curr_dark_pixel - predicted_GH)
                            index = i
                    return index

                elif transformation == "c=a-b":
                    mindiff = 1000
                    predicted_GH = darkpixelG - darkpixelH
                    index = -1
                    l=[]
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        print curr_dark_pixel
                        if abs(curr_dark_pixel - predicted_GH) < mindiff:
                            mindiff = abs(curr_dark_pixel - predicted_GH)
                            index=i
                        elif abs(curr_dark_pixel - predicted_GH) == mindiff:
                            upper = darkpixelupperG - darkpixelupperH
                            curr_index = problem.figures[str(index)]
                            I = problem.figures[str(i)]
                            mindiff_upper = abs(self.dark_pixel_upperhalf(curr_index) - upper)
                            mindiff_i = abs(self.dark_pixel_upperhalf(I) - upper)
                            if mindiff_i<mindiff_upper:
                                index = i
                            elif mindiff_i == mindiff_upper:
                                lower = darkpixelowerG - darkpixelowerH
                                curr_index = problem.figures[str(index)]
                                I = problem.figures[str(i)]
                                mindiff_lower = abs(self.dark_pixel_lowerhalf(curr_index) - lower)
                                mindiff_i_l = abs(self.dark_pixel_lowerhalf(I) - lower)
                                if mindiff_i_l < mindiff_lower:
                                    index = i

                    return index

                elif transformation == "half minus":
                    predicted_Iupper = abs(darkpixelupperG - darkpixelowerH)
                    predicted_Ilower = abs(darkpixelowerG - darkpixelupperH)
                    mindiff = 1000
                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_upper = self.dark_pixel_upperhalf(curr)
                        curr_dark_lower = self.dark_pixel_lowerhalf(curr)
                        if abs(curr_dark_upper - predicted_Iupper) + abs(curr_dark_lower - predicted_Ilower) < mindiff:
                            index = i
                            mindiff = abs(curr_dark_upper - predicted_Iupper) + abs(curr_dark_lower - predicted_Ilower)
                    return index

                elif transformation == "half-half":
                    print darkpixelupperG
                    print darkpixelowerH
                    mindiff = 1000
                    predicted_GH = darkpixelupperG + darkpixelowerH
                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel_upperhalf(curr) + self.dark_pixel_lowerhalf(curr)
                        print str(self.dark_pixel_upperhalf(curr)) + "up-low" + str(self.dark_pixel_lowerhalf(curr))
                        if abs(curr_dark_pixel - predicted_GH) < mindiff:
                            mindiff = abs(curr_dark_pixel - predicted_GH)
                            index = i
                    return index

                elif transformation == "xor":
                    mindiff = 1000

                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        print curr_dark_pixel
                        if curr_dark_pixel / float(darkpixelG) ==1:
                            continue
                        img_i = Image.open((problem.figures[str(i)]).visualFilename).convert('1')
                        val = self.isxor(g_xor_h, img_i)
                        print "xor ratio is " + str(val)
                        if abs(val - 1) < mindiff:
                            mindiff = abs(val - 1)
                            index = i
                    return index

                elif transformation == "center_and":
                    mindiff = 1000

                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        print curr_dark_pixel

                        img_i = Image.open((problem.figures[str(i)]).visualFilename).convert('1')
                        g_and_i = ImageChops.logical_and(img_g, img_i)
                        val = self.isand(g_and_i, img_h)
                        print "and ratio is " + str(val)
                        if abs(val - 1) < mindiff:
                            mindiff = abs(val - 1)
                            index = i
                    return index

                elif transformation == "and":
                    mindiff = 1000

                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        print curr_dark_pixel
                        if curr_dark_pixel / float(darkpixelG) ==1:
                            continue
                        img_i = Image.open((problem.figures[str(i)]).visualFilename).convert('1')
                        val = self.isand(g_and_h, img_i)
                        print "and ratio is " + str(val)
                        if abs(val - 1) < mindiff:
                            mindiff = abs(val - 1)
                            index = i
                    return index


                elif transformation == "special and":
                    mindiff = 1000

                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        print curr_dark_pixel

                        img_i = Image.open((problem.figures[str(i)]).visualFilename).convert('1')
                        b_and_i = ImageChops.logical_and(img_b, img_i)
                        val = self.isand(b_and_i, img_d)
                        print "and ratio is " + str(val)
                        if abs(val - 1) < mindiff:
                            mindiff = abs(val - 1)
                            index = i
                    return index

                elif transformation == "or":
                    mindiff = 1000

                    index = -1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        curr_dark_pixel = self.dark_pixel(curr)
                        print curr_dark_pixel
                        if curr_dark_pixel / float(darkpixelG) ==1:
                            continue
                        img_i = Image.open((problem.figures[str(i)]).visualFilename).convert('1')
                        val = self.isor(g_or_h, img_i)
                        print "and ratio is " + str(val)
                        if abs(val - 1) < mindiff:
                            mindiff = abs(val - 1)
                            index = i
                    return index



                elif transformation == "unsure":
                    visualAB = self.find_dark_ratio(A, B)
                    visualBC = self.find_dark_ratio(B, C)
                    visualGH = self.find_dark_ratio(G, H)
                    visualCF = self.find_dark_ratio(C, F)
                    if visualAB == 0 or visualBC == 0:
                        value = 0
                    else:
                        value = visualBC / visualAB
                    index = -1
                    best_score = 1
                    for i in range(1, 9):
                        curr = problem.figures[str(i)]
                        visualHcurr = self.find_dark_ratio(H, curr)
                        visualFcurr = self.find_dark_ratio(F, curr)
                        if visualGH == 0 or visualHcurr == 0:
                            value2_1 = 0
                        else:
                            value2_1 = visualHcurr / visualGH
                        if visualCF == 0 or visualFcurr == 0:
                            value2_2 = 0
                        else:
                            value2_2 = visualFcurr / visualCF
                        if abs(value2_1 - value) + abs(value2_2 - value) < best_score:
                            best_score = abs(value2_1 - value) + abs(value2_2 - value)
                            index = i
                    if index != -1:
                        print "the best score is " + str(best_score)
                        return index
                    else:
                        return -1

                return -1

    def get_diff(self, figure1, figure2):

        # if the length of figure1_objects and figure2_objects are different, it means additional objects or deletion of objects
        num_changed = len(figure2.objects) - len(figure1.objects)
        # between figure 1 and 2, check how many objects are added or deleted
        differences = {}
        differences["addition"] = {}
        differences["deleted"] = {}
        if num_changed == 0:
            differences["addition"] = 0
            differences["deleted"] = 0
        elif num_changed < 0:
            differences["addition"] = 0
            differences["deleted"] = abs(num_changed)
        else:
            differences["addition"] = abs(num_changed)
            differences["deleted"] = 0
        # compare each object between figure1 and figure2, use i to track the number of object
        i = 0
        figure1_objects = sorted(figure1.objects, key=figure1.objects.get)
        figure2_objects = sorted(figure2.objects, key=figure2.objects.get)

        for object1, object2 in zip(figure1_objects, figure2_objects):

            differences[i] = {}
            attributes1 = figure1.objects[object1].attributes
            attributes2 = figure2.objects[object2].attributes
            # get the keys from attributes1 and attributes2
            keys1 = set(attributes1.keys())
            keys2 = set(attributes2.keys())
            # find the changed attributes
            changed_attributes = []
            for key in keys1:
                if key in keys2 and attributes1[key] != attributes2[key]:
                    changed_attributes.append(key)
            # find what are the changes
            for x in changed_attributes:
                if x == 'shape':
                    differences[i][x] = attributes1['shape'] + " to " + attributes2['shape']

                if x == 'size':
                    differences[i][x] = self.object_size[attributes1['size']] - self.object_size[attributes2['size']]

                if x == 'fill':
                    differences[i][x] = attributes1['fill'] + " to " + attributes2['fill']

                if x == 'angle':
                    angle1 = int(attributes1['angle'])
                    angle2 = int(attributes2['angle'])
                    if 90 - angle1 == angle2 - 90 or 270 - angle1 == angle2 - 270:
                        differences[i][x] = "Ysymmetry"
                    elif 180 - angle1 == angle2 - 180 or 360 - angle1 == angle2 - 360:
                        differences[i][x] = "Xymmetry"
                    else:
                        differences[i][x] = angle1 - angle2
                if x == 'alignment':
                    if attributes1['alignment'] == 'bottom-right' and attributes2['alignment'] == 'bottom-left':
                        differences[i][x] = 'y_symmetry'
                    elif attributes1['alignment'] == 'bottom-left' and attributes2['alignment'] == 'bottom-right':
                        differences[i][x] = 'y_symmetry'
                    elif attributes1['alignment'] == 'top-left' and attributes2['alignment'] == 'top-right':
                        differences[i][x] = 'y_symmetry'
                    elif attributes1['alignment'] == 'top-right' and attributes2['alignment'] == 'top-left':
                        differences[i][x] = 'y_symmetry'
                    elif attributes1['alignment'] == 'bottom-left' and attributes2['alignment'] == 'top-left':
                        differences[i][x] = 'x_symmetry'
                    elif attributes1['alignment'] == 'top-left' and attributes2['alignment'] == 'bottom-left':
                        differences[i][x] = 'x_symmetry'
                    elif attributes1['alignment'] == 'top-right' and attributes2['alignment'] == 'bottom-left':
                        differences[i][x] = 'x_symmetry'
                    elif attributes1['alignment'] == 'bottom-right' and attributes2['alignment'] == 'top-right':
                        differences[i][x] = 'x_symmetry'
                    elif attributes1['alignment'] == attributes2['alignment']:
                        differences[i][x] = 'unchanged'

                if x == 'inside':

                    if len(attributes1[x]) > len(attributes2[x]):

                        differences[i][x] = 'added_inside'
                    elif len(attributes1[x]) < len(attributes2[x]):

                        differences[i][x] = 'removed_inside'

                if x == 'height':
                    differences[i][x] = self.object_height[attributes1['height']] - self.object_size[
                        attributes2['height']]

                if x == 'width':
                    differences[i][x] = self.object_width[attributes1['width']] - self.object_size[attributes2['width']]

                if x == 'above':
                    differences[i][x] = "above other object"

            i += 1
            # collect all the differences between figure1 and figure2
        empty_keys = [k for k, v in differences.iteritems() if not v]
        for k in empty_keys:
            del differences[k]

        print figure1.name + " compare to " + figure2.name
        print differences

        return differences

        # get visual score

    def visual_score(self, figure1, figure2):
        image1 = Image.open(figure1.visualFilename).convert('L')
        image2 = Image.open(figure2.visualFilename).convert('L')
        h1 = image1.histogram()
        h2 = image2.histogram()
        rms = math.sqrt(reduce(operator.add, map(lambda x, y: (x - y) ** 2, h1, h2)) / len(h1))
        return rms

    def find_dark_ratio(self, figure1, figure2):
        image1 = Image.open(figure1.visualFilename)
        image2 = Image.open(figure2.visualFilename)
        load1 = image1.load()
        load2 = image2.load()
        pixels1 = []
        for i in range(0, image1.size[0]):
            for j in range(0, image1.size[1]):
                pixels1.append(load1[i, j])

        pixels2 = []
        for i in range(0, image2.size[0]):
            for j in range(0, image2.size[1]):
                pixels2.append(load2[i, j])

        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (255, 255, 255, 255):
                nblack1 += 1
        ratio1 = nblack1

        nblack2 = 0
        for i in range(0, len(pixels2)):
            pixel = pixels2[i]
            if pixel == (255, 255, 255, 255):
                nblack2 += 1
        ratio2 = nblack2

        return float(ratio2) / float(ratio1)

    def dark_pixel(self, figure1):
        image1 = Image.open(figure1.visualFilename)
        load1 = image1.load()
        pixels1 = []
        for i in range(0, image1.size[0]):
            for j in range(0, image1.size[1]):
                pixels1.append(load1[i, j])
        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (0, 0, 0, 255):
                nblack1 += 1
        return nblack1

    def dark_pixel_upperhalf(self, figure1):
        image = Image.open(figure1.visualFilename)
        area = (0, 0, image.size[1], image.size[0] / 2)
        image1 = image.crop(area)
        load1 = image1.load()
        pixels1 = []
        for i in range(0, image1.size[0]):
            for j in range(0, image1.size[1]):
                pixels1.append(load1[i, j])
        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (0, 0, 0, 255):
                nblack1 += 1
        return nblack1

    def dark_pixel_lowerhalf(self, figure1):
        image = Image.open(figure1.visualFilename)
        area = (0, image.size[0] / 2, image.size[0], image.size[1])
        image1 = image.crop(area)
        load1 = image1.load()
        pixels1 = []
        for i in range(0, image1.size[0]):
            for j in range(0, image1.size[1]):
                pixels1.append(load1[i, j])
        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (0, 0, 0, 255):
                nblack1 += 1
        return nblack1

    def get_xor_darkpixel(self, figure1, figure2):
        image1 = Image.open(figure1.visualFilename).convert('1')
        image2 = Image.open(figure2.visualFilename).convert('1')
        image1 = image1.resize((image1.size[0], image1.size[1]), Image.NEAREST)
        image2 = image2.resize((image1.size[0], image1.size[1]), Image.NEAREST)

        image3 = ImageChops.logical_xor(image1, image2)
        image = image3.convert('RGBA')
        load1 = image.load()
        pixels1 = []
        for i in range(0, image.size[0]):
            for j in range(0, image.size[1]):
                pixels1.append(load1[i, j])
        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (0, 0, 0, 255):
                nblack1 += 1
        return nblack1

    def get_xor_ratio(self, figure1, figure2, figure3):
        image1 = Image.open(figure1.visualFilename).convert('1')
        image2 = Image.open(figure2.visualFilename).convert('1')
        image3 = Image.open(figure3.visualFilename).convert('1')

        image2 = image2.resize((image1.size[0], image1.size[1]), Image.NEAREST)
        image3 = image3.resize((image1.size[0], image1.size[1]), Image.NEAREST)

        pixels1 = []
        image = image3.convert('RGBA')
        load1 = image.load()
        for i in range(0, image.size[0]):
            for j in range(0, image.size[1]):
                pixels1.append(load1[i, j])
        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (0, 0, 0, 255):
                nblack1 += 1
        nblack2 = self.get_xor_darkpixel(figure1, figure2)
        if float(nblack2) == 0:
            return 0
        
        return float(nblack1) / float(nblack2)

    def isxor(self, image1, image2):
        pixels1 = image1.getdata()
        pixels2 = image2.getdata()

        len1 = len(pixels1)
        len2 = len(pixels2)

        black1 = 0
        black2 = 0
        for pixel in pixels1:
            if pixel == 0:
                black1 += 1
        black1_xor = len1 - black1
        for pixel in pixels2:
            if pixel == 0:
                black2 += 1
        if float(black1_xor) == 0:
            return 0
        return float(black2) / float(black1_xor)

    def isand(self, image1, image2):
        pixels1 = image1.getdata()
        pixels2 = image2.getdata()

        len1 = len(pixels1)
        len2 = len(pixels2)

        black1 = 0
        black2 = 0
        for pixel in pixels1:
            if pixel == 0:
                black1 += 1

        for pixel in pixels2:
            if pixel == 0:
                black2 += 1
        if float(black1) == 0:
            return 0
        return float(black2) / float(black1)

    def isor(self, image1, image2):
        pixels1 = image1.getdata()
        pixels2 = image2.getdata()

        len1 = len(pixels1)
        len2 = len(pixels2)

        black1 = 0
        black2 = 0
        for pixel in pixels1:
            if pixel == 0:
                black1 += 1

        for pixel in pixels2:
            if pixel == 0:
                black2 += 1
        if float(black2) == 0:
            return 0

        return float(black1) / float(black2)

    def ispartialxor(self, image1, image2):
        pixels1 = list(image1.getdata())[:len(list(image1.getdata())) / 3]
        pixels2 = list(image2.getdata())[:len(list(image2.getdata())) / 3]

        len1 = len(pixels1)
        len2 = len(pixels2)

        black1 = 0
        black2 = 0
        for pixel in pixels1:
            if pixel == 255:
                black1 += 1

        for pixel in pixels2:
            if pixel == 0:
                black2 += 1
        if black2 == 0:
            return 0

        return float(black1) / float(black2)



    def dark_pixel_lefthalf(self, figure1):
        image = Image.open(figure1.visualFilename)
        area = (0, 0, image.size[0]/2, image.size[1])
        image1 = image.crop(area)
        load1 = image1.load()
        pixels1 = []
        for i in range(0, image1.size[0]):
            for j in range(0, image1.size[1]):
                pixels1.append(load1[i, j])
        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (0, 0, 0, 255):
                nblack1 += 1
        return nblack1

    def dark_pixel_righthalf(self, figure1):
        image = Image.open(figure1.visualFilename)
        area = (image.size[0]/2, 0, image.size[0], image.size[1])
        image1 = image.crop(area)
        load1 = image1.load()
        pixels1 = []
        for i in range(0, image1.size[0]):
            for j in range(0, image1.size[1]):
                pixels1.append(load1[i, j])
        nblack1 = 0
        for i in range(0, len(pixels1)):
            pixel = pixels1[i]
            if pixel == (0, 0, 0, 255):
                nblack1 += 1
        return nblack1





