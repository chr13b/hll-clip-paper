// UltraLogLog raw-estimate generator.
// Uses Dynatrace hash4j's production UltraLogLog (the reference implementation;
// no Python bindings exist). Real hashing: each distinct key (a sequential long)
// is hashed with wyhashFinal3 (long->long) before insertion, exactly as a
// production caller would. Duplicates never change the sketch, so as in the
// Python experiments L enters only the (Python-side) estimators; here we only
// emit RAW estimates per (n) cell. Trials use disjoint key ranges => independent.
//
// Usage: java -cp hash4j.jar:. UllDriver <p> <T> <seedOffset> <n1,n2,...> [fgra|mle]
// Output (stdout): one CSV line "n,estimate" per trial.

import com.dynatrace.hash4j.distinctcount.UltraLogLog;
import com.dynatrace.hash4j.hashing.Hasher64;
import com.dynatrace.hash4j.hashing.Hashing;

public class UllDriver {
    public static void main(String[] args) {
        int p = Integer.parseInt(args[0]);
        int T = Integer.parseInt(args[1]);
        long seedOffset = Long.parseLong(args[2]);
        String[] nStrs = args[3].split(",");
        UltraLogLog.Estimator est = UltraLogLog.DEFAULT_ESTIMATOR;  // OPTIMAL_FGRA
        if (args.length > 4 && args[4].equals("mle")) {
            est = UltraLogLog.MAXIMUM_LIKELIHOOD_ESTIMATOR;
        }
        Hasher64 hasher = Hashing.wyhashFinal3();

        StringBuilder sb = new StringBuilder();
        long key = seedOffset;
        for (String nStr : nStrs) {
            int n = Integer.parseInt(nStr.trim());
            for (int t = 0; t < T; t++) {
                UltraLogLog ull = UltraLogLog.create(p);
                for (int i = 0; i < n; i++) {
                    ull.add(hasher.hashLongToLong(key));
                    key++;
                }
                double estimate = ull.getDistinctCountEstimate(est);
                sb.append(n).append(',').append(estimate).append('\n');
            }
            // flush per n to bound memory
            System.out.print(sb);
            sb.setLength(0);
        }
    }
}
