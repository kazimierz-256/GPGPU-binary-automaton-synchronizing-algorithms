﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public struct Problem
{
    public int[] stateTransitioningMatrixA;
    public int[] stateTransitioningMatrixB;
    public int size;

    public static Problem GenerateWorstCase(int n)
    {
        if (n < 3)
            throw new Exception("wrong number!");

        var problem = new Problem
        {
            size = n,
            stateTransitioningMatrixA = Enumerable.Range(0, n)
            .Select(i => i == 0 ? 1 : i)
            .ToArray(),
            stateTransitioningMatrixB = Enumerable.Range(0, n)
            .Select(i => (i + 1) == n ? 0 : (i + 1))
            .ToArray()
        };
        return problem;
    }

    // TODO: change this to fill arrray of problems
    public static void FillArrayOfProblems(Problem[] problems, int problemCount, int problemSize, int seed)
    {
        var random = new Random(seed);
        for (int i = 0; i < problemCount; i++)
        {
            problems[i] = GenerateProblem(problemSize, random.Next());
        }
    }

    public static Problem GenerateProblem(int problemSize, int seed)
    {
        var random = new Random(seed);
        var problem = new Problem
        {
            stateTransitioningMatrixA = new int[problemSize],
            stateTransitioningMatrixB = new int[problemSize],
            size = problemSize
        };

        for (int i = 0; i < problemSize; i++)
        {
            problem.stateTransitioningMatrixA[i] = random.Next(0, problemSize);
            problem.stateTransitioningMatrixB[i] = random.Next(0, problemSize);
        }

        return problem;
    }

    public override string ToString()
    {
        var builder = new StringBuilder("[[");
        for (int i = 0; i < size; i++)
        {
            builder.Append(stateTransitioningMatrixA[i]);
            if (i < size - 1)
                builder.Append(",");
        }
        builder.Append("],[");
        for (int i = 0; i < size; i++)
        {
            builder.Append(stateTransitioningMatrixB[i]);
            if (i < size - 1)
                builder.Append(",");
        }
        builder.Append("]]");
        return builder.ToString();
    }
}

