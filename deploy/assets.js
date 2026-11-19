import crypto from "crypto";

// Use the same normalized URI that Nginx uses to select the static file.
function authorized(request) {
    const secret = process.env.NOTEBOOK_ASSETS_LINK_SECRET;
    const signature = request.variables.arg_sig;
    if (!secret || !signature || !/^[0-9a-f]{64}$/.test(signature)) return "0";
    const expected = crypto.createHmac("sha256", secret)
        .update(request.rawVariables.uri).digest("hex");
    let difference = 0;
    for (let i = 0; i < expected.length; i++) {
        difference |= signature.charCodeAt(i) ^ expected.charCodeAt(i);
    }
    return difference === 0 ? "1" : "0";
}

export default { authorized };
