using System.Net.Mail;
using System.Security;
using System.Text;

namespace Notifications
{
    public class SendEmail
    {
        public void Notify(Problem problem, string email, SecureString password)
        {

            var client = new SmtpClient
            {
                Port = 587,
                Host = "smtp.office365.com",
                EnableSsl = true,
                Timeout = 10000,
                DeliveryMethod = SmtpDeliveryMethod.Network,
                UseDefaultCredentials = false,
                Credentials = new System.Net.NetworkCredential(email, password),
            };
            var mm = new MailMessage(
                email,
                email,
                "Cerny Cojecture",
                $@"This automata violates Cerny's conjecture: <a href='https://k256.bitbucket.io/cadcam/index.html?a="
                    + problem.ToString()
                    + $"' target='_blank'>{problem.ToString()}</a>"
                )
            {
                BodyEncoding = Encoding.UTF8,
                DeliveryNotificationOptions = DeliveryNotificationOptions.OnFailure,
                IsBodyHtml = true
            };

            client.Send(mm);
        }
    }
}
